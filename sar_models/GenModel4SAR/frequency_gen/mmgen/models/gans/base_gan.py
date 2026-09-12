from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn


class BaseGAN(nn.Module):
    """Shared loss parsing helpers for the Pix2Pix training loop."""

    @property
    def with_gen_auxiliary_loss(self):
        return (hasattr(self, 'gen_auxiliary_losses')
                and self.gen_auxiliary_losses is not None)

    @property
    def with_disc_auxiliary_loss(self):
        return (hasattr(self, 'disc_auxiliary_losses')
                and self.disc_auxiliary_losses is not None)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(item.mean() for item in loss_value)
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars.items() if 'loss' in key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
