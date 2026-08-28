"""维度 7 · 完整性与元数据（6 项）。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from ..base import MetricResult, Status
from ..profiles import _c0
from ..modules.metadata_rules import run_rules, summarize

DIM = "完整性与元数据"


def compute(ctx) -> list[MetricResult]:
    return [
        _coverage(ctx),
        _fill_consistency(ctx),
        _field_completeness(ctx),
        _calibration_traceability(ctx),
        _mask_layer(ctx),
        _card4l(ctx),
    ]


def _coverage(ctx):
    d = _c0(ctx.sar.intensity)
    valid = np.isfinite(d) & (d > 0)
    cov = float(valid.mean()) * 100.0
    status = Status.PASS if cov > 95 else (Status.WARN if cov > 80 else Status.FAIL)
    return MetricResult(
        key="integrity.coverage", name="有效数据覆盖率", dimension=DIM,
        value=cov, unit="%", status=status,
        threshold="有效像元 / 标称幅宽",
        reason=f"有效像元占比 {cov:.2f}%。",
    )


def _fill_consistency(ctx):
    d = _c0(ctx.sar.intensity)
    invalid = ~np.isfinite(d) | (d <= 0)
    frac = float(invalid.mean()) * 100.0
    if invalid.any():
        lbl, n = label(invalid)
        sizes = np.bincount(lbl.ravel())
        sizes = sizes[1:] if len(sizes) > 1 else np.array([0])
        largest = float(sizes.max()) / d.size * 100.0 if sizes.size else 0.0
    else:
        largest = 0.0
    if frac < 0.5:
        status, note = Status.PASS, "无效值占比可忽略。"
    elif largest > 1.0:
        status, note = Status.FAIL, "存在大块无效区且未被标记为填充（缺产品掩膜）。"
    else:
        status, note = Status.WARN, "存在无效区，请确认产品已正确标记填充值。"
    return MetricResult(
        key="integrity.fill_consistency", name="无效值 / 填充标识", dimension=DIM,
        value=frac, unit="%", status=status,
        threshold="掩膜 vs 实际无效区一致性（B 未标记 / D 已标记）",
        reason=note + "（无产品掩膜时以 0/NaN 区作代理）",
        detail={"largest_cc_pct": largest},
    )


def _field_completeness(ctx):
    results = run_rules(ctx.sar.metadata if ctx.sar.metadata else None)
    s = summarize(results)
    if s["threshold_total"] == 0:
        return MetricResult(key="integrity.field_completeness", name="元数据字段齐备性", dimension=DIM,
                            status=Status.NODATA, threshold="对照 CEOS-ARD Threshold / Goal 条款",
                            reason="未提供元数据边车。")
    achieved = s["threshold_achieved"]
    status = Status.PASS if achieved == s["threshold_total"] else Status.WARN
    return MetricResult(
        key="integrity.field_completeness", name="元数据字段齐备性", dimension=DIM,
        value=achieved, unit=f"/ {s['threshold_total']}", status=status,
        threshold="对照 CEOS-ARD Threshold / Goal 条款逐条自评",
        reason=f"Threshold 级 {achieved}/{s['threshold_total']} 项达成。",
        detail={"threshold_achieved": achieved, "threshold_total": s["threshold_total"],
                "goal_achieved": s["goal_achieved"], "goal_total": s["goal_total"]},
    )


def _calibration_traceability(ctx):
    md = ctx.sar.metadata
    if not md:
        return MetricResult(key="integrity.calibration_traceability", name="定标常数可追溯", dimension=DIM,
                            status=Status.NODATA, threshold="定标因子 / 辅助数据 / 处理器版本审计",
                            reason="未提供元数据，无法审计版本台账。")
    keys = ["calibration_factor", "calibration_constant", "sigma0_lut",
            "processor_version", "software_version", "auxiliary_version"]
    have = [k for k in keys if k in md]
    status = Status.PASS if len(have) >= 2 else Status.WARN
    return MetricResult(
        key="integrity.calibration_traceability", name="定标常数可追溯", dimension=DIM,
        status=status, threshold="定标因子 / 辅助数据 / 处理器版本审计",
        reason=f"已提供 {len(have)} 项可追溯字段：{', '.join(have) or '无'}。",
        detail={"provided_fields": have},
    )


def _mask_layer(ctx):
    md = ctx.sar.metadata
    if not md:
        return MetricResult(key="integrity.mask_layer", name="掩膜与不确定度层", dimension=DIM,
                            status=Status.NODATA, threshold="是否提供逐像素质量层",
                            reason="未提供元数据，无法判断是否含逐像素质量 / 掩膜层。")
    has_mask = any(k in md for k in ("mask", "quality_layer", "pixel_mask", "annotation"))
    return MetricResult(
        key="integrity.mask_layer", name="掩膜与不确定度层", dimension=DIM,
        status=Status.PASS if has_mask else Status.WARN,
        threshold="是否提供逐像素质量 / 掩膜 / 不确定度层（ARD 核心要求）",
        reason="已提供逐像素质量层。" if has_mask else "未提供逐像素质量 / 掩膜层。",
    )


def _card4l(ctx):
    results = run_rules(ctx.sar.metadata if ctx.sar.metadata else None)
    s = summarize(results)
    if s["threshold_total"] == 0:
        return MetricResult(key="integrity.card4l", name="CARD4L 合规等级", dimension=DIM,
                            status=Status.NODATA, threshold="逐条自评 Threshold / Goal",
                            reason="未提供元数据边车，无法逐条自评。")
    th_ratio = s["threshold_achieved"] / max(s["threshold_total"], 1)
    status = Status.PASS if th_ratio == 1.0 else (Status.WARN if th_ratio >= 0.6 else Status.FAIL)
    return MetricResult(
        key="integrity.card4l", name="CARD4L 合规等级", dimension=DIM,
        status=status, threshold="逐条自评 Threshold / Goal（可追溯性优先于绝对精度）",
        reason=f"Threshold {s['threshold_achieved']}/{s['threshold_total']}、Goal {s['goal_achieved']}/{s['goal_total']} 达成。",
    )
