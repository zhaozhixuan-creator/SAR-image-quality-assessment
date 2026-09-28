"""方位角一致性损失（专利向：质检辅助估计器 R 作为「冻结教师」监督生成器）。

背景
----
v3 质检系统用辅助网络 R(·)（AspectAngleEstimator，v2 gan_metrics.models，论文 §3.1.4）
在真实 SAR 图上训练、评估时冻结：把生成图的 (sin, cos) 反推为方位角并计算 CMAE。
质检画像显示 stylegan 的 CMAE≈91°（近似随机），即生成器当前忽略了条件方位角。

机制
----
本模块把同一个 R(·) 作为冻结的方位角教师，在 StyleGAN2 训练的 Gmain 阶段对生成图施加
方位角一致性损失：R(gen) 的 (sin,cos) 方向要等于条件标签中的 (sin,cos) 方向。

这使「质检定位弱点（CMAE 高）→ 定向修复（方位角损失微调）→ 指标改善」的闭环成立，
是『质检不但说哪里差，还指导往哪改』的落地机制。

加载兼容性
----------
R_128.pt 由 v3 评估环境（torch2.4）torch.save 的完整模型；本训练环境 torch1.7 经实测
可跨版本加载（pickle 需要 AspectAngleEstimator 在 `gan_metrics.models` 下可 import，
故在此按原名注册，不依赖 v2 运行时路径）。
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin: int, cout: int, pool: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class AspectAngleEstimator(nn.Module):
    """R(·)：128×128 单通道 → (sin φ, cos φ)，与 v2 gan_metrics.models 完全一致。"""

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


def load_azimuth_estimator(path: str, device) -> nn.Module:
    """加载冻结的方位角教师 R(·)：eval + requires_grad_(False)，梯度只流向生成器输入。"""
    # pickle 要求 class 位于 `gan_metrics.models`，按原名注册同名模块即可跨环境加载。
    parent = types.ModuleType("gan_metrics")
    mod = types.ModuleType("gan_metrics.models")
    mod.AspectAngleEstimator = AspectAngleEstimator
    sys.modules.setdefault("gan_metrics", parent)
    sys.modules["gan_metrics.models"] = mod

    R = torch.load(path, map_location="cpu")
    R.eval()
    R.requires_grad_(False)
    return R.to(device)
