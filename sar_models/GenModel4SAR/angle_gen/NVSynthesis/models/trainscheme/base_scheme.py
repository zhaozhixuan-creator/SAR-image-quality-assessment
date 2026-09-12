import logging
import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel

import core.scheduler.lr_scheduler as lr_scheduler
import models.networks.networks as networks
from models.losses.loss import MatchingLoss


logger = logging.getLogger("base")


class BaseScheme:
    def __init__(self, opt, device=None):
        self.opt = opt
        self.device = device or torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")

        self.model = networks.define_network(opt).to(self.device)
        self.model = DataParallel(self.model, device_ids=opt["gpu_ids"]).to(self.device)

        self.is_train = opt["is_train"]
        if self.is_train:
            train_opt = opt["setting"]
            self.model.train()
            self.log_dict = OrderedDict()
            self.init_optimizers(train_opt)
            self.init_schedulers(self.optimizer, train_opt)
            self.loss_fn = MatchingLoss(**train_opt["criterion"]).to(self.device)

    def init_optimizers(self, train_opt):
        optim_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optim_params.append(param)
            else:
                logger.warning("Params [{:s}] will not optimize.".format(name))

        optimizer = train_opt["optimizer"]
        weight_decay = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                optim_params,
                lr=train_opt["lr_G"],
                weight_decay=weight_decay,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                optim_params,
                lr=train_opt["lr_G"],
                weight_decay=weight_decay,
                betas=(train_opt["beta1"], train_opt["beta2"]),
            )
        else:
            raise NotImplementedError("Optimizer [{:s}] is not supported.".format(optimizer))

    def init_schedulers(self, optimizer, train_opt):
        if train_opt["lr_scheme"] != "MultiStepLR":
            raise NotImplementedError("Only MultiStepLR is configured for this project.")

        self.scheduler = lr_scheduler.MultiStepLR_Restart(
            optimizer,
            train_opt["lr_steps"],
            restarts=train_opt["restarts"],
            weights=train_opt["restart_weights"],
            gamma=train_opt["lr_gamma"],
            clear_state=train_opt["clear_state"],
        )

    def Train_BP(self, **kwargs):
        loss = self.loss_fn(**kwargs)
        loss.backward()
        self.optimizer.step()
        self.log_dict["loss"] = loss.item()

    def save(self, epoch, current_step=None):
        if current_step is None:
            save_filename = "{}.pt".format(epoch)
        else:
            save_filename = "epoch_{}_iter_{}.pt".format(epoch, current_step)
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        torch.save(
            {
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "model": self.model.state_dict(),
                "step": current_step,
                "epoch": epoch,
            },
            save_path,
        )

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        self.scheduler.step()
        if cur_iter < warmup_iter:
            warmup_lrs = [
                param_group["initial_lr"] / warmup_iter * cur_iter
                for param_group in self.optimizer.param_groups
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lrs):
                param_group["lr"] = lr

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_current_log(self):
        return self.log_dict

    def resume_training(self, pretrain_pt_fileName):
        ckpt = torch.load(os.path.join(pretrain_pt_fileName), map_location=self.device)
        model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model_module = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_module, "load_checkpoint_state"):
            model_module.load_checkpoint_state(model_state, strict=False, logger=logger)
        else:
            self.model.load_state_dict(model_state)
