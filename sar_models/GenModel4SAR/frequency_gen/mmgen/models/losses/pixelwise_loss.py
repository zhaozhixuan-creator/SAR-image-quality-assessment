import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from .utils import weighted_loss

_REDUCTION_MODES = ['none', 'mean', 'sum', 'batchmean', 'flatmean']
_PDF_LOSS_TYPES = {
    'ot': 'ot',
    'transport': 'ot',
    'wasserstein': 'ot',
    'kl': 'kl',
    'kld': 'kl',
    'js': 'js',
    'jsd': 'js',
}


@weighted_loss
def l1_loss(pred, target, n_edge):
    target = target[:, :pred.shape[1], ...]
    if n_edge > 0:
        pred = pred[:, :, n_edge:-n_edge, n_edge:-n_edge]
        target = target[:, :, n_edge:-n_edge, n_edge:-n_edge]
    return F.l1_loss(pred, target, reduction='none')


def _canonical_pdf_loss_type(loss_type):
    loss_type = loss_type.lower()
    if loss_type not in _PDF_LOSS_TYPES:
        raise ValueError(
            f'Unsupported PDF loss type: {loss_type}. Supported ones are: '
            f'{sorted(_PDF_LOSS_TYPES)}')
    return _PDF_LOSS_TYPES[loss_type]


def _soft_pdf(values, bin_centers, sigma, eps):
    values = values.reshape(-1, 1)
    bin_centers = bin_centers.reshape(1, -1)
    hist = torch.exp(-0.5 * ((values - bin_centers) / sigma)**2).sum(dim=0)
    hist = hist + eps
    return hist / hist.sum()


def _get_bin_centers(pred, target, n_bins, value_range, sigma, eps):
    if n_bins <= 1:
        raise ValueError('n_bins must be larger than 1 for KL/JS PDF loss.')

    if value_range is None:
        lower = min(float(pred.detach().min()), float(target.detach().min()))
        upper = max(float(pred.detach().max()), float(target.detach().max()))
    else:
        if len(value_range) != 2:
            raise ValueError('value_range must be a sequence of length 2.')
        lower, upper = float(value_range[0]), float(value_range[1])

    if upper <= lower:
        lower -= 0.5
        upper += 0.5

    bin_width = (upper - lower) / n_bins
    sigma = bin_width if sigma is None else float(sigma)
    if sigma <= 0:
        raise ValueError('sigma must be larger than 0 for KL/JS PDF loss.')

    bin_centers = torch.linspace(
        lower + 0.5 * bin_width,
        upper - 0.5 * bin_width,
        n_bins,
        device=pred.device,
        dtype=pred.dtype)
    return bin_centers, pred.new_tensor(max(sigma, eps))


def _pdf_divergence(pred, target, loss_type, n_bins, value_range, sigma, eps):
    bin_centers, sigma = _get_bin_centers(
        pred, target, n_bins, value_range, sigma, eps)
    pred_pdf = _soft_pdf(pred, bin_centers, sigma, eps)
    target_pdf = _soft_pdf(target, bin_centers, sigma, eps)

    if loss_type == 'kl':
        return torch.sum(target_pdf * (torch.log(target_pdf) -
                                       torch.log(pred_pdf)))

    mixture_pdf = 0.5 * (pred_pdf + target_pdf)
    target_kl = torch.sum(target_pdf * (torch.log(target_pdf) -
                                        torch.log(mixture_pdf)))
    pred_kl = torch.sum(pred_pdf * (torch.log(pred_pdf) -
                                    torch.log(mixture_pdf)))
    return 0.5 * (target_kl + pred_kl)


@weighted_loss
def pdf_loss(pred,
             target,
             n_edge,
             width_patch,
             loss_type='ot',
             n_bins=64,
             value_range=(-1.0, 1.0),
             sigma=None,
             eps=1e-6):
    loss_type = _canonical_pdf_loss_type(loss_type)
    target = target[:, :pred.shape[1], ...]
    _, channels, height, width = target.shape
    row_start = n_edge
    row_stop = height - n_edge - width_patch + 1
    col_start = n_edge
    col_stop = width - n_edge - width_patch + 1
    n_rows = max(math.floor((height - 2 * n_edge) / width_patch), 0)
    n_cols = max(math.floor((width - 2 * n_edge) / width_patch), 0)
    n_patch = n_rows * n_cols
    if n_patch == 0:
        raise ValueError(
            'No PDF-loss patches fit inside the image. Check n_edge and '
            'width_patch.')

    loss = pred.new_tensor(0.0)
    for channel in range(channels):
        for row in range(row_start, row_stop, width_patch):
            row_end = min(height - n_edge, row + width_patch)
            for col in range(col_start, col_stop, width_patch):
                col_end = min(width - n_edge, col + width_patch)
                target_patch = target[:, channel, row:row_end,
                                      col:col_end].reshape(-1)
                pred_patch = pred[:, channel, row:row_end,
                                  col:col_end].reshape(-1)
                if loss_type == 'ot':
                    target_sorted, _ = torch.sort(target_patch)
                    pred_sorted, _ = torch.sort(pred_patch)
                    loss = loss + torch.mean((target_sorted - pred_sorted)**2)
                    continue

                loss = loss + _pdf_divergence(
                    pred_patch,
                    target_patch,
                    loss_type=loss_type,
                    n_bins=n_bins,
                    value_range=value_range,
                    sigma=sigma,
                    eps=eps)

    return loss / n_patch


class _MappedLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        avg_factor=None,
        data_info=None,
        loss_name='loss',
    ):
        super().__init__()
        if reduction not in _REDUCTION_MODES:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_REDUCTION_MODES}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.data_info = data_info
        self._loss_name = loss_name

    def _resolve_kwargs(self, args, kwargs):
        if self.data_info is None:
            return args, kwargs
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'A mapped loss expects the model output dictionary as input.')
            outputs_dict = args[0]
        elif 'outputs_dict' in kwargs:
            outputs_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parse arguments passed to this loss module.')
        mapped = {key: outputs_dict[value] for key, value in self.data_info.items()}
        kwargs.update(mapped)
        return (), kwargs

    def loss_name(self):
        return self._loss_name


@MODULES.register_module()
class L1Loss(_MappedLoss):
    def __init__(self, n_edge=32, loss_name='loss_l1', **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)
        self.n_edge = n_edge

    def forward(self, *args, **kwargs):
        args, kwargs = self._resolve_kwargs(args, kwargs)
        return l1_loss(
            *args,
            weight=self.loss_weight,
            reduction=self.reduction,
            avg_factor=self.avg_factor,
            n_edge=self.n_edge,
            **kwargs)


@MODULES.register_module()
class pdfLoss(_MappedLoss):
    def __init__(self,
                 n_edge=32,
                 width_patch=256,
                 loss_type='ot',
                 n_bins=64,
                 value_range=(-1.0, 1.0),
                 sigma=None,
                 eps=1e-6,
                 loss_name='loss_pdf',
                 **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)
        self.n_edge = n_edge
        self.width_patch = width_patch
        self.loss_type = _canonical_pdf_loss_type(loss_type)
        self.n_bins = n_bins
        self.value_range = value_range
        self.sigma = sigma
        self.eps = eps

    def forward(self, *args, **kwargs):
        args, kwargs = self._resolve_kwargs(args, kwargs)
        return pdf_loss(
            *args,
            weight=self.loss_weight,
            reduction=self.reduction,
            avg_factor=self.avg_factor,
            n_edge=self.n_edge,
            width_patch=self.width_patch,
            loss_type=self.loss_type,
            n_bins=self.n_bins,
            value_range=self.value_range,
            sigma=self.sigma,
            eps=self.eps,
            **kwargs)
