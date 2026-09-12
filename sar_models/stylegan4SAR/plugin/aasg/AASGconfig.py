from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class AASGConfig:
    """Configuration for the AASG plugin.

    This config is intentionally standalone so it can be imported from both
    training and generation code paths without touching legacy modules.
    """

    # 是否启用 AASG。False 时应退化为原始主干行为（由接入侧保证）。
    enabled: bool = False

    # 条件向量 c 的布局约定: c = [类别 onehot..., sin(theta), cos(theta)]。
    # class_dim: 类别 onehot 维度。
    class_dim: int = 5
    # angle_dim: 角度分支维度（通常固定为 2，对应 sin/cos）。
    angle_dim: int = 2
    # cond_angle_start: 角度分支在 c 中的起始下标。
    cond_angle_start: int = 5

    # 稀疏散射先验参数。
    # k_points: 稀疏散射点数量 K。
    k_points: int = 16
    # base_resolution: 稀疏图 S0 的基础分辨率（H=W）。
    base_resolution: int = 32
    # sigma: 软栅格化高斯核宽度，值越大散射点越“模糊”。
    sigma: float = 0.08
    # uv_radius_scale: 将 tanh 输出再缩放到更小中心区域，避免散射点跑到四角。
    # 例如 0.6 表示 uv 最远只到 [-0.6, 0.6]，而不是 [-1, 1]。
    uv_radius_scale: float = 0.6
    # mlp_hidden_dim: AASG 内部 MLP 隐层宽度（w1 映射与点参数预测共用）。
    mlp_hidden_dim: int = 256

    # 角度可微变换参数。
    # scale_clamp: 对 log-scale 进行截断的范围，避免缩放发散。
    # 建议收紧到较小范围，避免几何变换把能量甩出画幅。
    scale_clamp: float = 0.25
    # use_class_adaptive_scale: 是否启用“按类别可变”的缩放偏置。
    use_class_adaptive_scale: bool = False

    # 中低层注入配置。
    # inject_resolutions: 需要注入 AASG 先验的分辨率层。
    inject_resolutions: tuple = (8, 16, 32, 64)
    # inject_alphas: 各分辨率对应的固定注入权重（建议与分辨率同步递减）。
    inject_alphas: tuple = (0.12, 0.08, 0.05, 0.03)
    # inject_warmup_kimg: 注入权重线性 warmup 时长（kimg）。
    inject_warmup_kimg: float = 1500.0

    # amp_eps: 散射点幅值下限，防止点幅值整体塌缩为 0。
    amp_eps: float = 1e-5

    # 可视化配置。
    # viz_enabled: 是否允许导出中间可视化（如 S0 / S_theta）。
    viz_enabled: bool = True
    # viz_max_items: 单次可视化最多保存的样本数上限。
    viz_max_items: int = 8

    def to_dict(self) -> Dict[str, Any]:
        # 将 dataclass 转成字典，便于日志/覆盖配置/序列化使用。
        return asdict(self)


DEFAULT_AASG_CONFIG = AASGConfig()


def build_aasg_config(overrides: Optional[Dict[str, Any]] = None) -> AASGConfig:
    """根据默认配置构建 AASGConfig，可选传入 overrides 覆盖部分字段。"""

    if overrides is None:
        return DEFAULT_AASG_CONFIG

    valid_keys = set(DEFAULT_AASG_CONFIG.to_dict().keys())
    unknown = [k for k in overrides.keys() if k not in valid_keys]
    if unknown:
        # 防止拼写错误静默生效，提升配置可靠性。
        raise KeyError(f"Unknown AASG config keys: {unknown}")

    cfg = DEFAULT_AASG_CONFIG
    for key, value in overrides.items():
        cfg = replace(cfg, **{key: value})

    if len(cfg.inject_resolutions) != len(cfg.inject_alphas):
        # 每个注入分辨率都必须有且仅有一个对应权重。
        raise ValueError("inject_resolutions and inject_alphas must have the same length")

    return cfg


def as_inject_alpha_map(inject_resolutions: Iterable[int], inject_alphas: Iterable[float]) -> Dict[int, float]:
    # 将 (resolutions, alphas) 转为 {res: alpha} 形式，便于主干按分辨率查表注入。
    return {int(r): float(a) for r, a in zip(inject_resolutions, inject_alphas)}
