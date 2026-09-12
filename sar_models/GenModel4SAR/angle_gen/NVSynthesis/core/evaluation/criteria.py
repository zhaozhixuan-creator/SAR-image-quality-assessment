import math

import numpy as np
import torch


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1_max = img1.max()
    img2_max = img2.max()
    if img1_max != 0:
        img1 = img1 / img1_max
    if img2_max != 0:
        img2 = img2 / img2_max
    return np.mean((img1 - img2) ** 2)


def calculate_normalized_mse(img1, img2, min_max=(-1, 1)):
    """Calculate MSE after mapping a known value range to [0, 1]."""

    img1 = _to_normalized_tensor(img1, min_max)
    img2 = _to_normalized_tensor(img2, min_max)
    return float(torch.mean((img1 - img2) ** 2).item())


def calculate_target_region_mse(img1, img2, min_max=(-1, 1), top_ratio=0.2):
    """Calculate MSE on the brightest target scatterer region of the GT image."""

    img1 = _to_normalized_tensor(img1, min_max)
    img2 = _to_normalized_tensor(img2, min_max)
    mask = _top_intensity_mask(img2, top_ratio)
    return float(torch.mean((img1[mask] - img2[mask]) ** 2).item())


def calculate_weighted_mse(img1, img2, min_max=(-1, 1), weight_power=1.0):
    """Calculate MSE weighted by GT intensity to reduce background dominance."""

    img1 = _to_normalized_tensor(img1, min_max)
    img2 = _to_normalized_tensor(img2, min_max)
    weights = torch.pow(img2, weight_power)
    weight_sum = torch.sum(weights)
    if weight_sum <= 0:
        return float(torch.mean((img1 - img2) ** 2).item())
    return float(torch.sum(weights * (img1 - img2) ** 2).item() / weight_sum.item())


def calculate_top_intensity_ratio(img, min_max=(-1, 1), top_ratio=0.2):
    img = _to_normalized_tensor(img, min_max)
    return float(_top_intensity_mask(img, top_ratio).float().mean().item())


def _to_normalized_tensor(img, min_max):
    if torch.is_tensor(img):
        img = img.detach().float().cpu()
    else:
        img = torch.from_numpy(np.asarray(img, dtype=np.float32))

    min_value, max_value = min_max
    img = torch.clamp(img, min_value, max_value)
    return (img - min_value) / (max_value - min_value)


def _top_intensity_mask(img, top_ratio):
    flat = img.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("Cannot calculate a target-region mask for an empty image.")

    top_ratio = float(top_ratio)
    if top_ratio <= 0 or top_ratio > 1:
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}.")

    k = max(1, int(math.ceil(flat.numel() * top_ratio)))
    threshold = torch.topk(flat, k, largest=True).values[-1]
    return img >= threshold
