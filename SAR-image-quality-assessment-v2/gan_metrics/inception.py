"""Inception-v3 特征提取器（FID 用）。

按 pytorch-fid 约定：取 Mixed_7c 输出经全局平均池化得到的 2048 维特征。
仅作 frozen 前向，不训练。若预训练权重不可用（离线 / 下载失败），回退随机初始化——
此时 FID 数值只用于验证计算流程，不具分布判别意义。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import gray_to_rgb


class InceptionV3(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            # 显式传 aux_logits / init_weights 会与权重加载冲突，用默认值即可
            # （IMAGENET1K_V1 权重按 aux_logits=True 保存）。
            self.model = inception_v3(weights=weights, transform_input=False)
        except Exception as exc:  # pragma: no cover - 离线 / 权重下载失败
            from torchvision.models import inception_v3
            self.model = inception_v3(weights=None, transform_input=False)
            print(f"[warn] Inception 预训练权重不可用，回退随机初始化（FID 仅供流程演示）：{exc}")
        self.model.eval()
        self._feat = None
        self.model.Mixed_7c.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 299, 299) float，范围 [0, 1]
        self._feat = None
        _ = self.model(x)
        f = F.adaptive_avg_pool2d(self._feat, (1, 1))
        return f.reshape(f.size(0), -1)  # (N, 2048)


def preprocess(images, size: int = 299, global_max: float | None = None) -> torch.Tensor:
    """图像集 → Inception 输入张量 (N, 3, size, size)。

    灰度→RGB，用全局最大值（跨集一致）归一化到 [0, 1]，再双线性 resize。
    global_max 须跨 real/fake 一致，故由调用方传入。
    """
    if global_max is None:
        global_max = float(max(np.max(gray_to_rgb(im)) for im in images))
    global_max = max(global_max, 1e-9)
    arrs = [gray_to_rgb(im).astype(np.float64) / global_max for im in images]
    t = torch.from_numpy(np.stack(arrs, axis=0)).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t
