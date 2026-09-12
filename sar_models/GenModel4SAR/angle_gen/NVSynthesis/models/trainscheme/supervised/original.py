import torch
from models.trainscheme.base_scheme import BaseScheme


class CustomizedScheme(BaseScheme):
    def __init__(self, opt, device=None):
        # optimizer, scheduler, network, loss are all initiated here
        super(CustomizedScheme, self).__init__(opt)
    
    def Train_OptimizeParameters(self, preinput_dict):
        input_dict = self.Train_PreProcess(preinput_dict)
        # function: scale
        self.optimizer.zero_grad()
        network_output = self.Network_Forward(input_dict)
        # function: get GT for BP
        self.Train_BP(**network_output)


    def Train_PreProcess(self, preinput_dict):
        """
        INPUT: preinput_dict
        """
        def add_noise_to_img(clean_img, logsnr, noise):
            alpha = logsnr.sigmoid().sqrt()
            sigma = (-logsnr).sigmoid().sqrt()
            alpha = alpha[:,None, None, None]
            sigma = sigma[:,None, None, None]
            return alpha * clean_img + sigma * noise

        noise = torch.randn_like(preinput_dict["x"])
        z_noisy = add_noise_to_img(preinput_dict["z"], preinput_dict["logsnr"], noise)

        B = noise.shape[0]
        cond_mask = (torch.rand((B,)) > self.opt["setting"]["cond_prob"])
        x_condition = torch.where(
                                    cond_mask[:, None, None, None], 
                                    preinput_dict["x"], 
                                    torch.randn_like(preinput_dict["x"])
                                )
        input_dict = {}
        input_dict["batch"] = {}
        input_dict["batch"]["logsnr"] = torch.stack([preinput_dict["logsnr_0"], preinput_dict["logsnr"]], dim=1).to(self.device)
        input_dict["batch"]["x"] = x_condition.to(self.device)
        input_dict["batch"]["z"] = z_noisy.to(self.device)
        input_dict["batch"]["azimuth_angle"] = preinput_dict["azimuth_angle"].to(self.device)
        input_dict["batch"]["incidence_angle"] = preinput_dict["incidence_angle"].to(self.device)
        input_dict["cond_mask"] = cond_mask.to(self.device)
        input_dict["GT_noise"] = noise.to(self.device)
        return input_dict
    
    def Network_Forward(self, input_dict):
        # return direct output from the network
        network_output = {}
        network_output["GT_noise"] = input_dict["GT_noise"]
        network_output["pred_noise"] = self.model(input_dict["batch"], cond_mask = input_dict["cond_mask"])
        return network_output

    def Test_OneIter(self, **preinput_dict):
        self.model.eval()
        with torch.no_grad():
            network_output = self.Network_Forward_test(preinput_dict)
        self.model.train()

        return network_output
    
    def Network_Forward_test(self, input_dict):
        # return direct output from the network
        network_output = {}
        network_output["x"] = input_dict["x"]
        network_output["z"] = input_dict["z"]
        network_output["azimuth_angle"] = input_dict["azimuth_angle"]
        network_output["incidence_angle"] = input_dict["incidence_angle"]

        B = input_dict["x"].shape[0]
        network_output["pred_noise"] = self.model(
            input_dict, cond_mask=torch.ones(B, dtype=torch.bool, device=self.device)
        ).detach().cpu()
        input_dict['x'] = torch.randn_like(input_dict['z'], device=self.device)
        network_output["pred_noise_unconditioned"] = self.model(
            input_dict, cond_mask=torch.zeros(B, dtype=torch.bool, device=self.device)
        ).detach().cpu()
        
        return network_output
        
