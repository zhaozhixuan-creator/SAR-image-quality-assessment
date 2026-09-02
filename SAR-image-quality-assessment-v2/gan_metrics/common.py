"""公用工具：目标/背景掩码、单通道取法、灰度→RGB、目标区裁剪。"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def to_2d(x: np.ndarray) -> np.ndarray:
    """取第一通道，兼容 (H, W) 与 (H, W, C)。"""
    x = np.asarray(x)
    if x.ndim == 3:
        return x[:, :, 0]
    return x


def target_background_masks(H: int, W: int, target_h: float = 0.4,
                            target_w: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """论文 §3.1.4 的固定目标/背景掩码。

    Mt：以图像中心为中心、边长 0.4H × 0.4W 的方形；Mb = 1 − Mt。
    论文对所有真实/生成图像共用同一掩码（可复现，避免掩码估计引入不稳定）。
    """
    th = int(round(H * target_h))
    tw = int(round(W * target_w))
    y0 = (H - th) // 2
    x0 = (W - tw) // 2
    target = np.zeros((H, W), dtype=bool)
    target[y0:y0 + th, x0:x0 + tw] = True
    return target, ~target


def gray_to_rgb(x: np.ndarray) -> np.ndarray:
    """单通道灰度复制为三通道（Inception 输入需要 RGB）。返回 (3, H, W)。"""
    g = to_2d(x)
    return np.stack([g, g, g], axis=0)


def crop_target(img: np.ndarray, target_h: float = 0.4, target_w: float = 0.4) -> np.ndarray:
    """裁出中心目标区（0.4H × 0.4W），供 E_asc 提取 ASC 感知特征。"""
    g = to_2d(img)
    H, W = g.shape
    th = int(round(H * target_h))
    tw = int(round(W * target_w))
    y0 = (H - th) // 2
    x0 = (W - tw) // 2
    return g[y0:y0 + th, x0:x0 + tw]
