"""维度 5 · 极化质量（5 项）。"""
from __future__ import annotations

import numpy as np

from ..base import MetricResult, Status

DIM = "极化质量"
EPS = 1e-12


def compute(ctx) -> list[MetricResult]:
    if not ctx.sar.is_polarimetric or ctx.sar.n_channels < 2:
        reason = "单通道图像，无极化通道，本维度指标不可评估。"
        return [
            MetricResult(key="polarimetric.amplitude_imbalance", name="通道幅度不平衡", dimension=DIM,
                         status=Status.NODATA, unit="dB", reason=reason,
                         threshold="S1C −0.5~+0.7 dB；LT-1A ≈ 0.2 dB"),
            MetricResult(key="polarimetric.phase_imbalance", name="通道相位不平衡", dimension=DIM,
                         status=Status.NODATA, unit="°", reason=reason, threshold="LT-1A ≈ 2°"),
            MetricResult(key="polarimetric.crosstalk", name="交叉极化串扰", dimension=DIM,
                         status=Status.NODATA, unit="dB", reason=reason,
                         threshold="LT-1A < −35 dB；ALOS-2 实测 −38 dB"),
            MetricResult(key="polarimetric.reciprocity", name="互易性 VH≈HV", dimension=DIM,
                         status=Status.NODATA, reason=reason, threshold="单站 SAR 应统计一致"),
            MetricResult(key="polarimetric.coregistration", name="通道间配准", dimension=DIM,
                         status=Status.NODATA, unit="像素", reason=reason),
        ]

    return [
        _amplitude_imbalance(ctx),
        _phase_imbalance(ctx),
        _crosstalk(ctx),
        _reciprocity(ctx),
        _coregistration(ctx),
    ]


def _intensity_ch(ctx, i):
    I = ctx.sar.intensity
    return I[:, :, i] if I.ndim == 3 else I


def _amplitude_imbalance(ctx):
    I = ctx.sar.intensity
    c0, c1 = I[:, :, 0], I[:, :, -1]  # 首末通道（dual: co/cross；quad: HH/VV）
    imb = float(10.0 * np.log10(np.maximum(c0.mean(), EPS) / np.maximum(c1.mean(), EPS)))
    status = Status.PASS if abs(imb) <= 0.7 else Status.WARN
    return MetricResult(
        key="polarimetric.amplitude_imbalance", name="通道幅度不平衡", dimension=DIM,
        value=imb, unit="dB", status=status,
        threshold="S1C −0.5~+0.7 dB；LT-1A ≈ 0.2 dB",
        reason=f"通道 {ctx.sar.channel_names[0]} vs {ctx.sar.channel_names[-1]} 均值比。",
    )


def _phase_imbalance(ctx):
    return MetricResult(
        key="polarimetric.phase_imbalance", name="通道相位不平衡", dimension=DIM,
        status=Status.NODATA, unit="°", threshold="LT-1A ≈ 2°",
        reason="需复数据（SLC）同极化相位差统计；幅度 / 强度图不含相位信息。",
    )


def _crosstalk(ctx):
    return MetricResult(
        key="polarimetric.crosstalk", name="交叉极化串扰", dimension=DIM,
        status=Status.NODATA, unit="dB", threshold="LT-1A < −35 dB；ALOS-2 实测 −38 dB",
        reason="需角反射器或已知极化特性分布目标反演串扰矩阵，单图无此类目标。",
    )


def _reciprocity(ctx):
    I = ctx.sar.intensity
    if I.ndim != 3 or I.shape[2] < 4:
        return MetricResult(
            key="polarimetric.reciprocity", name="互易性 VH≈HV", dimension=DIM,
            status=Status.NODATA, threshold="单站 SAR 应统计一致",
            reason="需四极化（VH 与 HV 两交叉极化通道），当前通道数不足。",
        )
    vh, hv = I[:, :, 2], I[:, :, 1]
    ratio = float(10.0 * np.log10(np.maximum(vh.mean(), EPS) / np.maximum(hv.mean(), EPS)))
    corr = float(np.corrcoef(vh.ravel(), hv.ravel())[0, 1])
    status = Status.PASS if abs(ratio) <= 1.0 and corr > 0.8 else Status.WARN
    return MetricResult(
        key="polarimetric.reciprocity", name="互易性 VH≈HV", dimension=DIM,
        value=ratio, unit="dB", status=status,
        threshold="单站 SAR 应统计一致（比值 ≈ 0 dB，相关系数 → 1）",
        reason=f"VH/HV 均值比 {ratio:.2f} dB，逐像素相关系数 {corr:.3f}。",
        detail={"corr": corr},
    )


def _coregistration(ctx):
    I = ctx.sar.intensity
    c0, c1 = I[:, :, 0].astype(np.float64), I[:, :, -1].astype(np.float64)
    dy, dx = _phase_corr_offset(c0, c1)
    mag = float(np.hypot(dy, dx))
    status = Status.PASS if mag < 0.5 else Status.WARN
    return MetricResult(
        key="polarimetric.coregistration", name="通道间配准", dimension=DIM,
        value=mag, unit="像素", status=status,
        threshold="通道间互相关偏移（应 < 0.5 像素）",
        reason=f"通道间偏移 (dx={dx:.2f}, dy={dy:.2f}) 像素。",
        detail={"dx": dx, "dy": dy},
    )


def _phase_corr_offset(a, b):
    """FFT 相位相关求通道间偏移（整数 + 质心细化）。"""
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= (np.abs(R) + EPS)
    cc = np.abs(np.fft.ifft2(R))
    cc = np.fft.fftshift(cc)
    y, x = np.unravel_index(np.argmax(cc), cc.shape)
    dy = y - cc.shape[0] // 2
    dx = x - cc.shape[1] // 2
    return float(dy), float(dx)
