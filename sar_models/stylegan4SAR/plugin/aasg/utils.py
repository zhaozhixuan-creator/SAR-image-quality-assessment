from typing import Tuple

import numpy as np
import torch


def normalize_map_for_viz(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize [N,1,H,W] or [N,C,H,W] tensor into [0,1] for visualization."""

    x = x.detach().float()
    dims = tuple(range(1, x.ndim))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """Convert one tensor image in CHW format into uint8 HWC."""

    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    x = x.detach().float().cpu().clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()
