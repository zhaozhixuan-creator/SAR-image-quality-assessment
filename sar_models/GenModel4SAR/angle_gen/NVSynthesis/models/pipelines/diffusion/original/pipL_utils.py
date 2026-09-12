# -*- coding:utf-8 -*-
import torch
import random
from tqdm import tqdm
import numpy as np
from models.pipelines.diffusion.base import BasePiPL

class DenoisingPiPL(BasePiPL):
    ################################################################################
    ###                             Initialization                               ###
    ################################################################################

    def __init__(self, opt, device=None):
        """
        Initialize pipeline with configuration options.

        Args:
            opt (dict): Parsed configuration dictionary.
            device (torch.device, optional): CUDA or CPU device.
        """
        super().__init__(opt, device)
        self.w = torch.tensor(self.opt["pipL"]["w"])

    def logsnr_schedule_cosine(self, t):
        """
        Define the SNR schedule for SDE
        """
        b = np.arctan(np.exp(-.5 * self.opt["pipL"]["logsnr_max"]))
        a = np.arctan(np.exp(-.5 * self.opt["pipL"]["logsnr_min"])) - b
        return -2. * torch.log(torch.tan(a * t + b))

    ################################################################################
    ###                             Train_Pipeline                               ###
    ################################################################################
    def Train_Pipeline(self, **input_from_dataloader):
        """
        Training step: generate input states, optimize, and log.

        Args:
            input_from_dataloader: Includes 'GT', 'current_step', etc.
        """
        preinput_dict = self.generate_train_state(**input_from_dataloader)
        self.scheme.Train_OptimizeParameters(preinput_dict)
        self.scheme.update_learning_rate(
            input_from_dataloader["current_step"], warmup_iter = self.opt["setting"]["warmup_iter"]
        ) 

        if input_from_dataloader["current_step"] % self.opt["logger"]["print_freq"] == 0:
            logs = self.scheme.get_current_log()
            message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                input_from_dataloader["epoch"], input_from_dataloader["current_step"], 
                self.scheme.get_current_learning_rate()
            )
            for k, v in logs.items():
                message += "{:s}: {:.4e} ".format(k, v)
            self.logger.info(message)
    
    def generate_train_state(self, **input_from_dataloader):
        """
        Generate noisy inputs for training from clean GT.

        Inputs:
        {
            "img": imgs, # SAR image pair (Bx2xCxHxW tensor)
            "incidence_angle": incidence_angle, # incidence angle in Radian (Bx2)
            "azimuth_angle": azimuth_angle, # azimuth angle in Radian (Bx2)
        }

        Returns:
            dict with keys: logsnr, logsnr_0, etc.
        """
        
        preinput_dict = {}
        img = input_from_dataloader["img"]
        B = img.shape[0] # number of images per batch
        x = img[:, 0] # (BxCxHxW tensor)
        z = img[:, 1] # (BxCxHxW tensor)

        # Randomly Select a random state
        timesteps = torch.rand((B,)) # To DO: is it needed to change to shape (input_from_dataloader["GT"].shape[0], 1, 1, 1)
        logsnr = self.logsnr_schedule_cosine(timesteps)
        logsnr_0 = self.logsnr_schedule_cosine(torch.zeros_like(logsnr))

        # generate the corresponding data
        preinput_dict["logsnr"] = logsnr
        preinput_dict["logsnr_0"] = logsnr_0
        preinput_dict["x"] = x
        preinput_dict["z"] = z
        preinput_dict["azimuth_angle"] = input_from_dataloader["azimuth_angle"]
        preinput_dict["incidence_angle"] = input_from_dataloader["incidence_angle"]
            
        return preinput_dict

    ################################################################################
    ###                              Test_Pipeline                               ###
    ################################################################################

    def Test_Pipeline(self, **input_from_dataloader):
        # init the testing iteration
        preinput_dict = self.generate_test_state_init(**input_from_dataloader)
        
        # start testing iteration
        for t in tqdm(reversed(range(0, self.T)), total=self.T, desc='diffusion loop', position=1, leave=False):
            output_dict = self.scheme.Test_OneIter(**preinput_dict)
            preinput_dict = self.generate_test_state_medium(t = t, **output_dict)

        # end testing iteration
        return preinput_dict
    
    def generate_test_state_init(self, **input_from_dataloader):
        """
        Generate data for testing iterations

        Inputs: input_from_dataloader
        {
            "img": imgs, # SAR image pair (Bx2xCxHxW tensor)
            "incidence_angle": incidence_angle, # incidence angle in Radian (Bx2)
            "azimuth_angle": azimuth_angle, # azimuth angle in Radian (Bx2)
        }
        """
        
        target_azimuth_angle = input_from_dataloader["azimuth_angle"][:, 1].unsqueeze(1) # Bx1
        target_incidence_angle = input_from_dataloader["incidence_angle"][:, 1].unsqueeze(1) # Bx1

        condition_img, condition_azimuth_angle, condition_incidence_angle = random.choice(self.record)
        azimuth_angle = torch.cat([condition_azimuth_angle, target_azimuth_angle], 1) # Bx2
        incidence_angle = torch.cat([condition_incidence_angle, target_incidence_angle], 1) # Bx2
        
        if azimuth_angle.shape[0] == 1:
            B = self.w.shape[0]
            azimuth_angle = azimuth_angle.repeat(B, 1) # Bx2
            incidence_angle = incidence_angle.repeat(B, 1) # Bx2
            condition_img = condition_img.repeat(B, 1, 1, 1) # BxCxHxW
        else:
            B = condition_img.shape[0]

        iterating_img = torch.randn_like(condition_img) # starting point of the iteration

        timesteps = torch.linspace(0., 1., self.T+1)[-1]
        logsnr = self.logsnr_schedule_cosine(timesteps-1)
        logsnr = logsnr.repeat(B)
        preinput_dict = {}
        preinput_dict["logsnr"] = torch.stack(
                                                [self.logsnr_schedule_cosine(torch.zeros_like(logsnr)), 
                                                logsnr], dim=1
                                            ).to(self.device)
        preinput_dict["x"] = condition_img.to(self.device)
        preinput_dict["z"] = iterating_img.to(self.device)
        preinput_dict["azimuth_angle"] = azimuth_angle.to(self.device)
        preinput_dict["incidence_angle"] = incidence_angle.to(self.device)
            
        return preinput_dict

    def generate_test_state_medium(self, t, **input_from_previous):
        """
        Generate data for testing iterations
           (intermedium iter): Transform the output_from_model to the format suitable for input_to_model
        """
        preinput_dict = {}
        target_azimuth_angle = input_from_previous["azimuth_angle"][:, 1].unsqueeze(1) # Bx1
        target_incidence_angle = input_from_previous["incidence_angle"][:, 1].unsqueeze(1) # Bx1

        condition_img, condition_azimuth_angle, condition_incidence_angle = random.choice(self.record)
        if condition_azimuth_angle.shape[0] == 1:
            B = self.w.shape[0]
            condition_azimuth_angle = condition_azimuth_angle.repeat(B, 1).to(self.device) # Bx2
            condition_incidence_angle = condition_incidence_angle.repeat(B, 1).to(self.device) # Bx2
            condition_img = condition_img.repeat(B, 1, 1, 1).to(self.device) # BxCxHxW
        else:
            B = condition_img.shape[0]
            condition_azimuth_angle = condition_azimuth_angle.to(self.device) # Bx2
            condition_incidence_angle = condition_incidence_angle.to(self.device) # Bx2
            condition_img = condition_img.to(self.device) # BxCxHxW
        azimuth_angle = torch.cat([condition_azimuth_angle, target_azimuth_angle], 1) # Bx2
        incidence_angle = torch.cat([condition_incidence_angle, target_incidence_angle], 1) # Bx2


        preinput_dict["x"] = condition_img.to(self.device)
        preinput_dict["azimuth_angle"] = azimuth_angle.to(self.device)
        preinput_dict["incidence_angle"] = incidence_angle.to(self.device)

        w = self.w.repeat(B)
        w = w[0:B]
        w = w[:, None, None, None]
        pred_noise_final = (1+w) * input_from_previous["pred_noise"] - w * input_from_previous["pred_noise_unconditioned"]
        z = input_from_previous["z"].detach().cpu()

        timesteps = torch.linspace(0., 1., self.T+1)[t+1]
        timesteps_nexts = torch.linspace(0., 1., self.T+1)[t]
        logsnr = self.logsnr_schedule_cosine(timesteps)
        logsnr_next = self.logsnr_schedule_cosine(timesteps_nexts)
        c = - torch.special.expm1(logsnr - logsnr_next)
        squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
        alpha, sigma, alpha_next = map(lambda x: x.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
            
        z_start = (z - sigma * pred_noise_final) / alpha
        z_start.clamp_(-1., 1.)
        
        model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)
        posterior_variance = squared_sigma_next * c

        if logsnr_next==0:
            update_z = model_mean
        else:
            update_z = model_mean + posterior_variance.sqrt() * torch.randn_like(z).cpu()
        preinput_dict["z"] = update_z.to(self.device)

        timesteps = torch.linspace(0., 1., self.T+1)[max(0, t)]
        logsnr = self.logsnr_schedule_cosine(timesteps)
        logsnr = logsnr.repeat(B)
        preinput_dict["logsnr"] = torch.stack(
                                                    [self.logsnr_schedule_cosine(torch.zeros_like(logsnr)), 
                                                    logsnr], dim=1
                                                ).to(self.device)

        return preinput_dict
