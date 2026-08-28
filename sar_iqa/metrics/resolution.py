"""维度 3 · 分辨率与点目标响应（5 项）。

依赖 ctx.irf（由 orchestrator 对自动检测的点目标做 IRF 过采样分析）。
无点目标时全部标记为无法评估。
"""
from __future__ import annotations

from ..base import MetricResult, Status

DIM = "分辨率与点目标"


def compute(ctx) -> list[MetricResult]:
    if not ctx.irf:
        note = "未在图像中检测到孤立强散射体（点目标），无法测量 IRF；请确保图像内含角反射器/强点。"
        return [
            MetricResult(key="resolution.resolution", name="空间分辨率（−3 dB）", dimension=DIM,
                         status=Status.NODATA, unit="m", reason=note,
                         threshold="16× 过采样 IRF 主瓣半功率宽度"),
            MetricResult(key="resolution.pslr", name="PSLR 峰值旁瓣比", dimension=DIM,
                         status=Status.NODATA, unit="dB", reason=note, threshold="加权后 < −20 dB"),
            MetricResult(key="resolution.islr", name="ISLR 积分旁瓣比", dimension=DIM,
                         status=Status.NODATA, unit="dB", reason=note, threshold="< −15 dB"),
            MetricResult(key="resolution.sslr", name="SSLR 次生旁瓣比", dimension=DIM,
                         status=Status.NODATA, unit="dB", reason=note),
            MetricResult(key="resolution.broadening", name="散焦 / 展宽因子", dimension=DIM,
                         status=Status.NODATA, reason=note, threshold="实测 / 理论主瓣宽度之比 ≈ 1"),
        ]

    irf = ctx.irf
    r = irf["range"]
    a = irf["azimuth"]
    sp_r = _spacing(ctx, "range")
    sp_a = _spacing(ctx, "azimuth")

    res_r_px = r["resolution_px"]
    res_a_px = a["resolution_px"]

    # 目标有效性校验：−3dB 宽度 < 1.2 像素 → 疑似单像素噪声尖峰，非真实点目标
    suspicious = min(res_r_px, res_a_px) < 1.2

    results = [
        _resolution(ctx, res_r_px, res_a_px, sp_r, sp_a),
        _side_metric(ctx, "resolution.pslr", "PSLR 峰值旁瓣比", "pslr_db",
                     "加权后 < −20 dB；未加权 sinc 基线 −13.3 dB（两者相差 7 dB，勿混用）", r, a),
        _side_metric(ctx, "resolution.islr", "ISLR 积分旁瓣比", "islr_db",
                     "加权 < −15 dB；未加权 sinc 基线 −9.97 dB", r, a),
        _side_metric(ctx, "resolution.sslr", "SSLR 次生旁瓣比", "sslr_db",
                     "非邻近旁瓣 / 主瓣", r, a),
        _broadening(ctx, res_r_px, res_a_px, sp_r, sp_a),
    ]
    if suspicious:
        note = "[目标存疑：−3dB 宽度 < 1.2 像素，疑似单像素噪声尖峰而非真实点目标] "
        for rr in results:
            rr.reason = note + rr.reason
            if rr.status == Status.PASS:
                rr.status = Status.WARN
    return results


def _spacing(ctx, axis):
    md = ctx.sar.metadata
    if axis == "range":
        return md.get("pixel_spacing_range") or md.get("pixel_spacing") or md.get("range_spacing")
    return md.get("pixel_spacing_azimuth") or md.get("pixel_spacing") or md.get("azimuth_spacing")


def _resolution(ctx, res_r, res_a, sp_r, sp_a):
    worst = max(res_r, res_a)
    detail = {"range_px": res_r, "azimuth_px": res_a}
    unit = "像素"
    if sp_r:
        detail["range_m"] = res_r * float(sp_r)
        unit = "m"
        if sp_a:
            detail["azimuth_m"] = res_a * float(sp_a)
    return MetricResult(
        key="resolution.resolution", name="空间分辨率（−3 dB）", dimension=DIM,
        value=worst, unit=unit, status=Status.PASS,
        threshold="16× 过采样 IRF 主瓣半功率宽度；标称 / 实测 / 像素间距三者不可混用",
        reason="由自动检测的点目标（%.0f, %.0f）IRF 测得。" % (ctx.point_targets[0][0], ctx.point_targets[0][1]),
        detail=detail,
    )


def _side_metric(ctx, key, name, field, threshold, r, a):
    vr, va = r[field], a[field]
    vals = [v for v in (vr, va) if v is not None]
    if not vals:
        return MetricResult(key=key, name=name, dimension=DIM, status=Status.NODATA,
                            unit="dB", threshold=threshold, reason="旁瓣能量过低，无法测量。")
    v = max(vals)
    # 阈值方向：PSLR/ISLR/SSLR 越小越好（更负越好）。
    # 关键：区分加权/未加权基线——未加权 sinc 理论 PSLR −13.3 dB / ISLR −9.97 dB，
    # 与加权后工程指标（−20 / −15 dB）相差甚远，直接套用会系统性误判（引用陷阱）。
    if field == "pslr_db":
        status = (Status.PASS if v <= -13.3 else
                  Status.WARN if v <= -10.0 else Status.FAIL)
    elif field == "islr_db":
        status = (Status.PASS if v <= -15.0 else
                  Status.WARN if v <= -9.97 else Status.FAIL)
    else:  # sslr_db 无硬性规范，仅作信息性判定
        status = Status.PASS if v <= -15.0 else Status.WARN
    return MetricResult(
        key=key, name=name, dimension=DIM, value=v, unit="dB", status=status,
        threshold=threshold,
        reason=f"距离向 {vr if vr is not None else float('nan'):.1f} dB / 方位向 {va if va is not None else float('nan'):.1f} dB。",
        detail={"range_db": vr, "azimuth_db": va},
    )


def _broadening(ctx, res_r, res_a, sp_r, sp_a):
    nominal = ctx.sar.metadata.get("nominal_resolution")
    if nominal is None:
        return MetricResult(
            key="resolution.broadening", name="散焦 / 展宽因子", dimension=DIM,
            status=Status.NODATA,
            threshold="实测 / 理论主瓣宽度之比 ≈ 1",
            reason="未提供标称分辨率（nominal_resolution）元数据，无法计算展宽因子。",
        )
    try:
        nominal = float(nominal)
    except (TypeError, ValueError):
        nominal = None
    if nominal is None or nominal <= 0:
        return MetricResult(key="resolution.broadening", name="散焦 / 展宽因子", dimension=DIM,
                            status=Status.NODATA, threshold="实测 / 理论主瓣宽度之比 ≈ 1",
                            reason="标称分辨率非法。")
    measured = max(res_r * float(sp_r or 1.0), res_a * float(sp_a or 1.0))
    factor = measured / nominal
    status = Status.PASS if factor < 1.25 else (Status.WARN if factor < 1.6 else Status.FAIL)
    return MetricResult(
        key="resolution.broadening", name="散焦 / 展宽因子", dimension=DIM,
        value=factor, unit="", status=status,
        threshold="实测 / 理论主瓣宽度之比 ≈ 1（显著 > 1 为聚焦退化）",
        reason=f"实测 {measured:.2f} m / 标称 {nominal:.2f} m。",
    )
