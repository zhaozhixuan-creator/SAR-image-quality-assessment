"""生成模型评估指标（论文《...》§3.1.4）。

实现六项指标，用于比较真实 SAR 图像集与生成图像集：
- 图像保真：FID、SSIM
- 目标区物理一致性：AFS
- 背景统计一致性：ΔENL、BVE
- 角度条件一致性：CMAE

与 sar_iqa（单图质检）相互独立，这里评估的是「生成模型」而非「单张图」。
"""
from .common import EPS, to_2d, target_background_masks, gray_to_rgb, crop_target
from .inception import InceptionV3
from .models import (
    AspectAngleEstimator,
    ASCExtractor,
    ASCAutoencoder,
    train_angle_estimator,
    train_asc_extractor,
)
from .metrics import fid, ssim, afs, delta_enl, bve, cmae, enl, estimate_angles

__all__ = [
    "fid", "ssim", "afs", "delta_enl", "bve", "cmae", "enl", "estimate_angles",
    "AspectAngleEstimator", "ASCExtractor", "ASCAutoencoder",
    "train_angle_estimator", "train_asc_extractor", "InceptionV3",
    "EPS", "to_2d", "target_background_masks", "gray_to_rgb", "crop_target",
]
