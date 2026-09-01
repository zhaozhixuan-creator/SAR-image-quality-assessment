"""分级质检流水线（对应 README §二）。

把 38 项指标按「原生产品级别」组织，落实「指标在其原生产品级别测量」原则：

    L0 原始级     下行完整性 / 丢包 / 坏像元
    L1/SLC 单视复 物理性能指标的唯一正确测量位置（IRF/NESZ/模糊/扇贝/极化）
    L2 地距级     几何与地形（ALE / RTC / 叠掩阴影 / 投影）
    L3+ 分析就绪  ARD 合规与相位质量（CARD4L / 相干性 / 干涉）

单图引擎默认把输入视为 L1/SLC 级；对非原生级的指标，若仍给出数值，
标记为「跨级测量」并附告警（跨级测得的数值不可与规格比对）。

编排顺序（§二 五层工程之「编排层」）：场景相关检查（前置筛查）先跑、
高成本点目标检查后跑。维度顺序见 metrics.DIMENSIONS（辐射/几何/分辨率/…），
点目标 IRF 仅在确有孤立强散射体时才过采样分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .io import SarImage
from .metrics import Context, run_all
from . import spec

# 默认产品级：单图引擎输入为已成像产品，按 SLC 级处理物理指标。
DEFAULT_LEVEL = spec.L1_SLC

# 各级标签（用于报告）
LEVEL_LABELS = {
    spec.L0: "L0 原始级",
    spec.L1_SLC: "L1/SLC 单视复级",
    spec.L2: "L2 地距/地理编码级",
    spec.L3: "L3+ 分析就绪级",
}


@dataclass
class PipelineResult:
    """一次分级质检的运行结果与分级汇总。"""

    results: list
    ctx: Context
    level: str
    native_keys: list[str] = field(default_factory=list)   # 原生级指标 key
    cross_keys: list[str] = field(default_factory=list)    # 跨级测得的指标 key
    level_counts: dict[str, int] = field(default_factory=dict)   # 各级指标总数（38 项全表）
    phase_counts: dict[int, int] = field(default_factory=dict)   # 各期指标总数
    level_evaluated: dict[str, int] = field(default_factory=dict)  # 各级已评估（非 NODATA）数

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "level_label": LEVEL_LABELS.get(self.level, self.level),
            "native_keys": self.native_keys,
            "cross_keys": self.cross_keys,
            "level_counts": self.level_counts,
            "phase_counts": self.phase_counts,
            "level_evaluated": self.level_evaluated,
        }


def resolve_level(level: str | None) -> str:
    """把 CLI 的 --level 解析为产品级；auto/None → 默认 SLC 级。"""
    if not level or level == "auto":
        return DEFAULT_LEVEL
    # 允许大小写 / 别名
    lv = level.upper()
    if lv in ("L0", "RAW"):
        return spec.L0
    if lv in ("L1", "SLC", "L1/SLC", "SINGLE"):
        return spec.L1_SLC
    if lv in ("L2", "GRD", "GEOCODED"):
        return spec.L2
    if lv in ("L3", "L3+", "ARD", "ANALYSIS"):
        return spec.L3
    # 未知值：原样返回并告警（由调用方处理）
    return level


def pipeline_summary(results, level: str) -> dict:
    """在已有结果上计算分级汇总（原生 / 跨级 / 各级评估数）。"""
    native, cross = [], []
    for r in results:
        s = spec.SPEC_BY_KEY.get(r.key)
        if s is None:
            continue
        if s.level == level:
            native.append(r.key)
        elif r.status.value != "nodata":
            # 跨级且给出了数值 → 需告警（跨级数值不可与规格比对）
            cross.append(r.key)
    evaluated_by_level: dict[str, int] = {lv: 0 for lv in spec.LEVELS}
    for r in results:
        s = spec.SPEC_BY_KEY.get(r.key)
        if s is not None and r.status.value != "nodata":
            evaluated_by_level[s.level] = evaluated_by_level.get(s.level, 0) + 1
    return {
        "level": level,
        "native_keys": native,
        "cross_keys": cross,
        "level_counts": spec.level_counts(),
        "phase_counts": spec.phase_counts(),
        "level_evaluated": evaluated_by_level,
    }


def run_pipeline(sar: SarImage, level: str | None = None, config: dict | None = None) -> PipelineResult:
    """分级质检入口：运行 38 项指标 + 自研模块依赖的共享上下文，返回分级汇总。"""
    lv = resolve_level(level)
    results, ctx = run_all(sar, config)
    summary = pipeline_summary(results, lv)
    return PipelineResult(results=results, ctx=ctx, **summary)
