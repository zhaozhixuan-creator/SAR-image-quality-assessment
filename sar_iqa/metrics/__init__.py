"""指标编排：运行全部 7 维 38 项指标，返回统一的结果列表。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..base import MetricResult
from ..io import SarImage
from ..point_target import analyze_irf, detect_point_targets
from ..profiles import col_profile, row_profile

from . import (radiation, geometric, resolution, noise, polarimetric,
               interferometric, integrity)


@dataclass
class Context:
    """一次质检运行的共享上下文：已加载图像 + 预计算剖面 + 点目标。"""

    sar: SarImage
    config: dict = field(default_factory=dict)
    row_prof: np.ndarray | None = None
    col_prof: np.ndarray | None = None
    point_targets: list = field(default_factory=list)
    irf: dict | None = None


# 各维度 compute 函数（按维度顺序）
DIMENSIONS = [
    radiation.compute,
    geometric.compute,
    resolution.compute,
    noise.compute,
    polarimetric.compute,
    interferometric.compute,
    integrity.compute,
]


def run_all(sar: SarImage, config: dict | None = None) -> tuple[list[MetricResult], Context]:
    """运行全部指标。

    返回 (results, context)。context 里保留剖面与点目标，供报告绘图与去斑模块使用。
    """
    config = config or {}
    ctx = Context(sar=sar, config=config)

    # 预计算剖面（线性功率域平均后转 dB）
    ctx.row_prof = row_profile(sar.intensity)
    ctx.col_prof = col_profile(sar.intensity)

    # 点目标检测（仅 SLC 级有效，这里在图像域自动找强点）
    n_pt = int(config.get("n_point_targets", 1))
    ctx.point_targets = detect_point_targets(sar.intensity, n=n_pt)
    ctx.irf = None
    if ctx.point_targets:
        y, x, _ = ctx.point_targets[0]
        ctx.irf = analyze_irf(
            sar.intensity, y, x,
            window=int(config.get("irf_window", 64)),
            factor=int(config.get("oversampling_factor", 16)),
        )

    results: list[MetricResult] = []
    for fn in DIMENSIONS:
        try:
            results.extend(fn(ctx))
        except Exception as e:  # 单项失败不阻断整体
            results.append(MetricResult(
                key=f"error.{fn.__module__}", name="内部错误",
                dimension="系统", reason=f"{type(e).__name__}: {e}",
            ))
    return results, ctx
