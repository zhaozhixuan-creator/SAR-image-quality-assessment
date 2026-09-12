from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ESFConfig:
    # 训练数据压缩包（由 data/ASCdata 打包得到）。
    dataset_zip: Path = Path("datasets/ascdata.zip")
    # 权重与日志输出目录。
    outdir: Path = Path("plugin/ESF_net/ascweight")

    # ASC 标签配置。
    k_points: int = 8
    # 属性维度: 4 -> [x_norm, y_norm, amp_A, alpha]
    attr_dim: int = 4

    # 训练超参数。
    epochs: int = 80
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    viz_interval: int = 10
    # K-fold 配置。k_folds<=1 时退化为普通 train/val。
    k_folds: int = 5

    # 网络结构与训练策略。
    base_channels: int = 32
    dropout: float = 0.1
    amp_loss_weight: float = 1.0
    alpha_loss_weight: float = 1.0
    xy_loss_weight: float = 1.0
    # 位置误差放大系数：先对 x/y 差值按该系数放大再算损失。
    # 对 128x128 图像，32~64 通常更合适。
    xy_error_scale: float = 64.0
    # 训练稳定性/梯度强度控制。
    loss_scale: float = 10.0
    grad_clip_norm: float = 5.0
    lr_scheduler_tmax: int = 80
    lr_scheduler_min: float = 1e-5
    # 随机种子。<0 表示不固定，每次运行随机。
    seed: int = -1

    # -----------------------------
    # 模块2注入判别器时使用的配置
    # -----------------------------
    # 预训练属性网络权重（冻结后注入 D）
    d_inject_ckpt: Path = Path("plugin/ESF_net/ascweight/esf_attr_best.pt")
    # ESF 分支 logit 融合权重
    d_fusion_weight: float = 0.3


DEFAULT_ESF_CONFIG = ESFConfig()
