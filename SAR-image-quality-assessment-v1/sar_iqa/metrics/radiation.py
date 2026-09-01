"""维度 1 · 辐射质量（6 项）。"""
from __future__ import annotations

import numpy as np

from ..base import MetricResult, Status
from ..profiles import detrend, dominant_period, find_steps, to_db, _c0

DIM = "辐射质量"


def compute(ctx) -> list[MetricResult]:
    return [
        _absolute_calibration(ctx),
        _stability(ctx),
        _eap_residual(ctx),
        _subband_consistency(ctx),
        _scalloping(ctx),
        _saturation(ctx),
    ]


def _absolute_calibration(ctx):
    return MetricResult(
        key="radiation.absolute_calibration", name="绝对辐射定标精度", dimension=DIM,
        status=Status.NODATA, unit="dB",
        threshold="工程门槛 1 dB(3σ)；S1C 实测 0.36 dB(1σ)、GF-3 1.3~1.4 dB",
        reason="需角反射器 RCS 反演 vs 理论 RCS，单张图像无已知 RCS 的目标，无法计算。",
    )


def _stability(ctx):
    return MetricResult(
        key="radiation.stability", name="辐射稳定性 / 长期漂移", dimension=DIM,
        status=Status.NODATA, unit="dB",
        threshold="LT-1A：<1 dB/1000 km、<0.5 dB/5 天",
        reason="需稳定分布目标（雨林）的 γ⁰ 时序，单张图像无时序数据。",
    )


def _eap_residual(ctx):
    prof = ctx.col_prof
    resid = detrend(prof, order=2)
    std = float(resid.std())
    p2p = float(np.percentile(resid, 95) - np.percentile(resid, 5))
    if std <= 0.5:
        status, note = Status.PASS, "距离向残差在容差内。"
    elif std <= 1.0:
        status, note = Status.WARN, "距离向残差偏大。"
    else:
        status, note = Status.FAIL, "距离向残差显著超差。"
    return MetricResult(
        key="radiation.eap_residual", name="跨轨天线方向图（EAP）残差", dimension=DIM,
        value=std, unit="dB", status=status,
        threshold="S1C 俯仰向增益变化 0.3 dB(1σ)；波位间 ≤ 0.2 dB",
        reason=note + "（注：单场景残差含地物贡献，无均匀场景校准时为筛查代理值）",
        detail={"p95_p5_db": p2p},
        raw={"col_profile": prof, "residual": resid},
    )


def _subband_consistency(ctx):
    prof = ctx.col_prof
    steps = find_steps(prof, thresh_db=0.5)
    max_step = float(abs(steps[0]["step_db"])) if steps else 0.0
    if max_step < 0.5:
        status, note = Status.PASS, "列均值剖面未检出子带台阶。"
    else:
        status, note = Status.WARN if max_step < 1.0 else Status.FAIL, \
            f"检出台阶 {max_step:.2f} dB @ 列 {steps[0]['position']}。"
    return MetricResult(
        key="radiation.subband_consistency", name="子带 / 拼接辐射一致性", dimension=DIM,
        value=max_step, unit="dB", status=status,
        threshold="子带交界处无可见台阶（<0.5 dB）",
        reason=note,
        detail={"n_steps": len(steps)},
        raw={"col_profile": prof, "steps": steps},
    )


def _scalloping(ctx):
    prof = ctx.row_prof
    resid = detrend(prof, order=1)
    dp = dominant_period(resid)
    p2p = dp["peak_to_peak"]
    if dp["period"] == 0 or p2p < 0.3:
        status, note = Status.PASS, "方位向行均值剖面无显著周期性起伏。"
    elif p2p < 0.6:
        status, note = Status.WARN, f"检测到扇贝起伏 {p2p:.2f} dB（周期约 {dp['period']:.0f} 行）。"
    else:
        status, note = Status.FAIL, f"扇贝起伏显著 {p2p:.2f} dB（周期约 {dp['period']:.0f} 行）。"
    return MetricResult(
        key="radiation.scalloping", name="扇贝效应残差", dimension=DIM,
        value=p2p, unit="dB", status=status,
        threshold="方位向行均值周期性起伏，峰峰值 / 标准差(dB)",
        reason=note,
        detail={"period_rows": dp["period"], "power_ratio": dp["power"]},
        raw={"row_profile": prof},
    )


def _saturation(ctx):
    d = _c0(ctx.sar.intensity)
    mx = float(d.max())
    if mx <= 0:
        return MetricResult(key="radiation.saturation", name="饱和率 / 动态范围", dimension=DIM,
                            status=Status.NODATA, unit="%", reason="图像全零。")
    # 直方图上界堆积：接近最大值的像元占比
    frac = float((d >= 0.999 * mx).mean()) * 100.0
    if frac < 0.5:
        status, note = Status.PASS, "无显著饱和堆积。"
    elif frac < 2.0:
        status, note = Status.WARN, "存在轻度上界堆积。"
    else:
        status, note = Status.FAIL, "上界堆积显著，疑似 ADC 削波 / 饱和。"
    return MetricResult(
        key="radiation.saturation", name="饱和率 / 动态范围", dimension=DIM,
        value=frac, unit="%", status=status,
        threshold="饱和率 → 缺陷类（B 类 / 重）",
        reason=note,
        raw={"hist_upper": frac, "max": mx},
    )
