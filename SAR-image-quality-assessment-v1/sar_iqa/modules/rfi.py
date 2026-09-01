"""自研模块 2 · RFI 谱域检测器（对应 docs/09）。

RFI 应在距离压缩前的谱域（距离频率—方位时间）做，特征最显著：
窄带 RFI = 贯穿方位向的亮竖线；宽带 RFI = 局部方位时段的水平亮块。

本模块提供：
- detect_rfi()      谱域检测（输入为原始回波，距离频率 x 方位时间），按 docs/09 伪代码实现；
- image_proxy()     图像域代理（单张已成像产品可用）：方位向行均值 3σ 检出亮条纹。
"""
from __future__ import annotations

import numpy as np


def detect_rfi(slc_raw: np.ndarray, noise_power: float | None = None) -> dict:
    """谱域 RFI 检测（需要距离压缩前的原始回波）。

    slc_raw: 原始回波（距离频率 x 方位时间），形状 (n_range_freq, n_azimuth)。
    返回 {mask, ratio, row_mask, col_mask}。
    """
    slc_raw = np.asarray(slc_raw, dtype=np.float64)
    spec = np.fft.fft(slc_raw, axis=1)          # 方位向 FFT
    power = np.abs(spec) ** 2

    row_mean = power.mean(axis=1)               # 距离频率维均值（窄带竖线）
    row_mask = row_mean > row_mean.mean() + 3 * row_mean.std()

    col_mean = power.mean(axis=0)               # 方位时间维均值（宽带水平块）
    col_mask = col_mean > col_mean.mean() + 3 * col_mean.std()

    rfi_mask = row_mask[:, None] | col_mask[None, :]
    ratio = float(rfi_mask.mean())
    return {
        "mask": rfi_mask,
        "ratio": ratio,
        "row_mask": row_mask,
        "col_mask": col_mask,
    }


def image_proxy(intensity: np.ndarray, sigma: float = 3.0) -> dict:
    """图像域 RFI 代理：方位向（行）均值 3σ 检出贯穿亮条纹。

    仅用于已成像产品；真实 RFI 检测应在谱域（见 detect_rfi）。
    返回 {ratio, affected_rows, row_profile, threshold}。
    """
    from ..profiles import _c0, to_db

    d = _c0(intensity)
    db = to_db(d)
    row = db.mean(axis=1)
    mu = float(row.mean())
    sd = float(row.std()) if row.size > 1 else 0.0
    thresh = mu + sigma * sd
    affected = (row > thresh).sum()
    ratio = float(affected / row.size)
    return {
        "ratio": ratio,
        "affected_rows": int(affected),
        "row_profile": row,
        "threshold": thresh,
    }
