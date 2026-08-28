"""一维剖面与统计工具：行/列均值、去趋势、周期检测、阶跃检测。

供辐射质量（扇贝 / 子带台阶 / EAP 残差）、噪声（丢行 / RFI 代理）等指标共用。
剖面在**线性功率域**平均后再转 dB，避免对 dB 直接做算术平均的物理偏差。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _c0(x: np.ndarray) -> np.ndarray:
    """取第一通道（单通道图即自身）。"""
    if x.ndim == 3:
        return x[:, :, 0]
    return x


def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), EPS))


def row_profile(intensity: np.ndarray) -> np.ndarray:
    """方位向（行）剖面，dB。对有效像元取中位数（抗点目标 / 边界填充污染）。"""
    d = _c0(intensity)
    return _robust_profile(d, axis=1)


def col_profile(intensity: np.ndarray) -> np.ndarray:
    """距离向（列）剖面，dB。对有效像元取中位数。"""
    d = _c0(intensity)
    return _robust_profile(d, axis=0)


def _robust_profile(d: np.ndarray, axis: int) -> np.ndarray:
    """有效像元（有限且 >0）的中位数剖面，填充 NaN 行/列用全局中位数兜底。"""
    import warnings

    valid = np.isfinite(d) & (d > 0)
    dm = np.where(valid, d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(dm, axis=axis)
    global_med = np.nanmedian(med)
    if not np.isfinite(global_med):
        global_med = 1.0
    med = np.where(np.isfinite(med), med, global_med)
    return to_db(med)


def detrend(x: np.ndarray, order: int = 1) -> np.ndarray:
    """多项式去趋势，返回残差（dB）。"""
    x = np.asarray(x, dtype=np.float64)
    idx = np.arange(len(x))
    poly = np.polynomial.polynomial.polyfit(idx, x, order)
    trend = np.polynomial.polynomial.polyval(idx, poly)
    return x - trend


def dominant_period(x: np.ndarray, min_period: int = 2, max_period: int | None = None):
    """用 FFT 找剖面中的主导周期分量（扇贝检测）。

    返回 dict: period(样本)、power(相对占比)、peak_to_peak(dB)。
    若无显著周期，period=0。
    """
    x = np.asarray(x, dtype=np.float64)
    r = x - x.mean()
    n = len(r)
    if n < 8:
        return {"period": 0, "power": 0.0, "peak_to_peak": 0.0}
    spec = np.abs(np.fft.rfft(r)) ** 2
    freqs = np.fft.rfftfreq(n)
    hi = min(max_period or n // 2, n // 2)
    lo = max(min_period, 2)
    valid = np.zeros_like(spec, dtype=bool)
    for i in range(1, len(freqs)):
        period = 1.0 / freqs[i] if freqs[i] > 0 else np.inf
        if lo <= period <= hi:
            valid[i] = True
    if not valid.any():
        return {"period": 0, "power": 0.0, "peak_to_peak": 0.0}
    i_best = int(np.argmax(np.where(valid, spec, 0.0)))
    period = 1.0 / freqs[i_best]
    total = spec[1:].sum() or 1.0
    power = spec[i_best] / total
    amp = 2.0 * np.abs(np.fft.rfft(r)[i_best]) / n
    return {"period": float(period), "power": float(power), "peak_to_peak": float(2 * amp)}


def find_steps(x: np.ndarray, thresh_db: float = 0.5, win: int = 5) -> list[dict]:
    """检测一维剖面中的阶跃（子带拼接台阶）。

    用中值滤波平滑后求相邻差分，超过 thresh_db 的位置记为阶跃。
    返回 [{position, step_db, direction}]，按 |step| 降序。
    """
    from scipy.ndimage import median_filter

    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return []
    smooth = median_filter(x, size=win)
    diff = np.diff(smooth)
    steps = []
    for i in range(len(diff)):
        if abs(diff[i]) >= thresh_db:
            steps.append({
                "position": i + 1,
                "step_db": float(diff[i]),
                "direction": "up" if diff[i] > 0 else "down",
            })
    steps.sort(key=lambda s: abs(s["step_db"]), reverse=True)
    return steps
