"""评估所用的两个小网络：ASC 提取器 E_asc 与方位角估计器 R(·)。

- R(·)：完全按论文 §3.1.4 实现——4 个卷积块（3×3 卷积 + BN + ReLU + 2×2 最大池化，
  通道 16/32/64/128）→ 全局平均池化 → 2 个全连接 → 输出 (sin φ, cos φ)。
  训练损失为式(43) 的正余弦 MSE，仅在真实图上训练、评估时冻结。

- E_asc：论文为「预训练且冻结的 ASC 提取器」，其训练协议未公开；这里用浅层卷积
  编码器（5 个卷积块 → GAP → dim 维）以「重构自监督」在真实目标切片上预训练，
  使特征对主导散射结构敏感，作为 ASC 感知特征的简化替身，支撑 AFS 余弦相似度。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin: int, cout: int, pool: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class AspectAngleEstimator(nn.Module):
    """R(·)：128×128 单通道 → (sin φ, cos φ)。"""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_ch, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)                         # (N, 128, 8, 8)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)   # (N, 128)
        return self.fc(f)                            # (N, 2)


class ASCExtractor(nn.Module):
    """E_asc 编码器：目标切片 → dim 维散射结构特征。"""

    def __init__(self, in_ch: int = 1, dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(in_ch, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, dim, pool=False),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)                          # (N, dim, H/16, W/16)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)   # (N, dim)
        return f


class ASCAutoencoder(nn.Module):
    """编码器 + 镜像解码器，用于以重构目标自监督预训练 E_asc。"""

    def __init__(self, in_ch: int = 1, dim: int = 128):
        super().__init__()
        self.encoder = ASCExtractor(in_ch, dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder.encoder(x))


def _to_tensor(arrays, size: int, normalize: bool = True) -> torch.Tensor:
    """list[(H, W)] → (N, 1, size, size) float 张量，逐图归一化到 [0, 1] 并 resize。"""
    imgs = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 3:
            a = a[:, :, 0]
        m = float(a.max())
        if normalize and m > 1e-9:
            a = a / m
        imgs.append(a)
    arr = np.stack(imgs, axis=0)[:, None, :, :]
    t = torch.from_numpy(arr)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t


def train_angle_estimator(images, angles_deg, epochs: int = 40, batch: int = 64,
                          lr: float = 1e-3, size: int = 128, device: str = "cpu",
                          seed: int = 0) -> AspectAngleEstimator:
    """在真实图像上训练 R(·)。images: list[(H, W)]，angles_deg: list[float]（度）。"""
    torch.manual_seed(seed)
    model = AspectAngleEstimator(1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = _to_tensor(images, size)
    rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    y = torch.from_numpy(np.stack([np.sin(rad), np.cos(rad)], axis=1)).float().to(device)
    n = X.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = model(X[idx].to(device))
            loss = F.mse_loss(out, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [R]     epoch {ep:2d}/{epochs}  mse={tot / n:.4f}")
    model.eval()
    return model


def train_asc_extractor(images, epochs: int = 20, batch: int = 64, lr: float = 1e-3,
                        size: int = 64, target_h: float = 0.4, target_w: float = 0.4,
                        device: str = "cpu", seed: int = 0, dim: int = 128) -> ASCExtractor:
    """以重构自监督在真实目标切片上预训练 E_asc，返回冻结的编码器。"""
    from .common import crop_target

    torch.manual_seed(seed)
    ae = ASCAutoencoder(1, dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    crops = [crop_target(im, target_h, target_w) for im in images]
    X = _to_tensor(crops, size)
    n = X.shape[0]
    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            out = ae(X[idx].to(device))
            loss = F.mse_loss(out, X[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [ASC]   epoch {ep:2d}/{epochs}  rec={tot / n:.4f}")
    ae.eval()
    return ae.encoder
