import torch.nn as nn
import torch.nn.functional as F
import einops

class MatchingLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Mapping of loss types to functions
        self.loss_fn = {
            'l1': self.l1_loss,
            'l2': self.l2_loss,
            'huber': self.huber_loss
        }
        
        self.weights = []
        self.loss_types = []
        
        for loss_name, loss_dict in kwargs.items():
            loss_type = loss_dict['loss_type']
            self.loss_types.append(loss_type)
            self.weights.append(loss_dict['weight'])
        
    def forward(self, **kwargs):
        loss = 0
        for idx, loss_fn in enumerate(self.loss_types):
            weight = self.weights[idx]
            if loss_fn in self.loss_fn:
                tmp_loss = self.loss_fn[loss_fn](**kwargs)
            loss += weight * tmp_loss
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean()
    
    def l1_loss(self, **kwargs):
        tmp_loss = 0
        GT_keys = [key for key in kwargs if "GT" in key] # GT_GT, GT_mean, GT_var/s
        for GT_name in GT_keys:
            pred_name = GT_name.replace("GT", "pred")
            if pred_name in kwargs:
                tmp_loss += F.l1_loss(kwargs[pred_name], kwargs[GT_name], reduction='none')
        return tmp_loss
    
    def l2_loss(self, **kwargs):
        tmp_loss = 0
        GT_keys = [key for key in kwargs if "GT" in key] # GT_GT, GT_mean, GT_var/s
        for GT_name in GT_keys:
            pred_name = GT_name.replace("GT", "pred")
            if pred_name in kwargs:
                tmp_loss += F.mse_loss(kwargs[pred_name], kwargs[GT_name], reduction='none')
        return tmp_loss
    
    def huber_loss(self, **kwargs):
        tmp_loss = 0
        GT_keys = [key for key in kwargs if "GT" in key] # GT_GT, GT_mean, GT_var/s
        for GT_name in GT_keys:
            pred_name = GT_name.replace("GT", "pred")
            if pred_name in kwargs:
                tmp_loss += F.smooth_l1_loss(kwargs[pred_name], kwargs[GT_name], reduction='none')
        return tmp_loss
