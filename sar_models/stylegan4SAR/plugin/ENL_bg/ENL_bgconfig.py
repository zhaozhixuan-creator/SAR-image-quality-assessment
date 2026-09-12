from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ENLBGConfig:
    """ENL_bg 配置类（模块3：背景一致性约束）。

    说明：
    - 所有 ENL_bg 超参数统一放在这里管理。
    - 训练侧只建议保留 `--enl_bg_enabled` 开关，其他参数从本配置读取。
    """

    # 是否启用 ENL_bg（默认关闭，避免影响已有基线流程）
    enabled: bool = False

    # ENL_bg 基础权重（最终有效权重会乘 warmup 系数）
    weight: float = 0.2

    # 线性 warmup 时长，单位 kimg；0 表示不 warmup
    warmup_kimg: float = 500.0

    # 背景掩码类型：
    # - center_box: 以图像中心矩形作为前景，外部为背景
    # - center_circle: 以图像中心圆形作为前景，外部为背景
    mask_type: str = 'center_box'

    # 前景区域比例（用于 center_box 或 center_circle）
    # center_box: 前景宽高 = fg_ratio * 原宽高
    # center_circle: 前景半径 = fg_ratio * min(H, W)
    fg_ratio: float = 0.4

    # ENL 计算中的方差稳定项，避免除零
    eps_var: float = 1e-6

    # log 域稳定项，避免 log(0)
    eps_log: float = 1e-6

    # 强度图转换方式：
    # - mean: 多通道取均值作为灰度强度
    intensity_mode: str = 'mean'

    # 调试开关：True 时 forward 返回更完整的中间统计
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """将配置转为字典，方便日志记录与覆盖。"""
        return asdict(self)


DEFAULT_ENLBG_CONFIG = ENLBGConfig()


def build_enlbg_config(overrides: Optional[Dict[str, Any]] = None) -> ENLBGConfig:
    """构建并校验 ENL_bg 配置。"""

    if overrides is None:
        cfg = DEFAULT_ENLBG_CONFIG
    else:
        valid_keys = set(DEFAULT_ENLBG_CONFIG.to_dict().keys())
        unknown = [k for k in overrides.keys() if k not in valid_keys]
        if unknown:
            raise KeyError(f"Unknown ENL_bg config keys: {unknown}")
        cfg = DEFAULT_ENLBG_CONFIG
        for key, value in overrides.items():
            cfg = replace(cfg, **{key: value})

    if cfg.weight < 0:
        raise ValueError('ENL_bg weight must be non-negative')
    if cfg.warmup_kimg < 0:
        raise ValueError('ENL_bg warmup_kimg must be non-negative')
    if cfg.mask_type not in ['center_box', 'center_circle']:
        raise ValueError("mask_type must be one of: ['center_box', 'center_circle']")
    if not (0.0 < cfg.fg_ratio < 1.0):
        raise ValueError('fg_ratio must be in (0, 1)')
    if cfg.eps_var <= 0 or cfg.eps_log <= 0:
        raise ValueError('eps_var and eps_log must be positive')
    if cfg.intensity_mode not in ['mean']:
        raise ValueError("intensity_mode currently supports: ['mean']")

    return cfg
