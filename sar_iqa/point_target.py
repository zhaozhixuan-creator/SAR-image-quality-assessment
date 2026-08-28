"""点目标检测与 IRF（脉冲响应）分析。

从单张图像中自动寻找孤立强散射体（无需角反射器已知坐标），
用 FFT 过采样到连续域后测 −3dB 分辨率、PSLR、ISLR、SSLR。
这是把维度 3（分辨率与点目标响应）在"无先验点目标"条件下可运行化的核心。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .profiles import _c0, to_db


def detect_point_targets(intensity: np.ndarray, n: int = 1, snr_db: float = 18.0,
                         min_sep: int = 24) -> list[tuple[int, int, float]]:
    """自动检测图像中最亮的孤立点目标。

    返回 [(y, x, peak_db), ...]，按峰值降序；找不到时返回空列表。
    """
    d = _c0(intensity)
    db = to_db(d)
    med = float(np.median(db))
    mad = float(np.median(np.abs(db - med)))
    sigma = 1.4826 * mad if mad > 0 else float(db.std())
    thresh = med + snr_db

    mask = db > thresh
    f = maximum_filter(db, size=5)
    local_max = (db == f) & mask
    ys, xs = np.nonzero(local_max)

    if len(ys) == 0:
        # 放宽：取最亮的 0.1% 作为候选
        thresh2 = float(np.percentile(db, 99.9))
        mask = db > thresh2
        local_max = (db == f) & mask
        ys, xs = np.nonzero(local_max)

    if len(ys) == 0:
        return []

    order = np.argsort(db[ys, xs])[::-1]
    candidates = [(int(ys[o]), int(xs[o]), float(db[ys[o], xs[o]])) for o in order]

    # 优先未饱和峰值：饱和（削波）点目标 IRF 已畸变，不适合测分辨率。
    # 一个角反射器通常不是全局最亮（可能有更亮的饱和舰船点），应跳过饱和候选。
    sat_level = float(np.percentile(db, 99.9))
    unsaturated = [c for c in candidates if c[2] < sat_level]
    ordered = unsaturated + [c for c in candidates if c[2] >= sat_level]

    picked: list[tuple[int, int, float]] = []
    for y, x, pk in ordered:
        if all(max(abs(y - py), abs(x - px)) > min_sep for py, px, _ in picked):
            picked.append((y, x, pk))
        if len(picked) >= n:
            break
    return picked


def oversample_window(win: np.ndarray, factor: int = 16) -> np.ndarray:
    """FFT 频域零填充实现整数倍过采样插值（连续域 IRF）。"""
    win = np.asarray(win, dtype=np.float64)
    H, W = win.shape
    F = np.fft.fftshift(np.fft.fft2(win))
    H2, W2 = H * factor, W * factor
    Fp = np.zeros((H2, W2), dtype=np.complex128)
    hh0, ww0 = H2 // 2 - H // 2, W2 // 2 - W // 2
    Fp[hh0:hh0 + H, ww0:ww0 + W] = F
    return np.fft.ifft2(np.fft.ifftshift(Fp)).real * (factor ** 2)


def _half_power_width(cut: np.ndarray, factor: int) -> float:
    """−3dB 主瓣宽度（幅度域，peak/√2），返回原像素单位，带亚像素插值。"""
    peak = float(cut.max())
    if peak <= 0:
        return 0.0
    half = peak / np.sqrt(2.0)
    idx = int(np.argmax(cut))
    # 左穿越点
    left = idx
    while left > 0 and cut[left] > half:
        left -= 1
    lw = _crossing(cut, left, left + 1, half)
    # 右穿越点
    right = idx
    while right < len(cut) - 1 and cut[right] > half:
        right += 1
    rw = _crossing(cut, right - 1, right, half)
    return (rw - lw) / factor


def _crossing(cut, i, j, level):
    """线性插值求剖面与 level 的交点横坐标。"""
    a, b = float(cut[i]), float(cut[j])
    if b == a:
        return float(i)
    return float(i) + (level - a) / (b - a) * (j - i)


def _mainlobe_bounds(cut: np.ndarray, idx: int) -> tuple[int, int]:
    """主瓣边界 = 峰值两侧的第一零点（第一个局部极小）。

    从峰向左/右走，直到剖面停止下降（开始上升），即第一零点。
    """
    n = len(cut)
    r = idx
    while r < n - 2 and cut[r] >= cut[r + 1]:
        r += 1
    l = idx
    while l > 1 and cut[l] >= cut[l - 1]:
        l -= 1
    return int(l), int(r)


def analyze_irf(intensity: np.ndarray, y: int, x: int, window: int = 64, factor: int = 16) -> dict:
    """分析单个点目标的 IRF，返回距离向/方位向的分辨率、PSLR、ISLR、SSLR。

    所有 dB 值基于幅度域（20·log10）。
    """
    d = _c0(intensity)
    H, W = d.shape
    y0 = max(0, y - window // 2)
    y1 = min(H, y + window // 2)
    x0 = max(0, x - window // 2)
    x1 = min(W, x + window // 2)
    win = np.sqrt(np.maximum(d[y0:y1, x0:x1], 0.0))  # 幅度
    if win.shape[0] < 8 or win.shape[1] < 8:
        return {}

    big = oversample_window(win, factor)
    # 峰值应在已知点目标位置附近（窗口中心）做局部搜索，而非全局 argmax，
    # 否则可能跳到窗内另一个更亮的散射体（如舰船 / 邻近距离目标）。
    cy, cx = y - y0, x - x0
    r_lo = max(0, (cy - 4) * factor)
    r_hi = min(big.shape[0], (cy + 4) * factor + 1)
    c_lo = max(0, (cx - 4) * factor)
    c_hi = min(big.shape[1], (cx + 4) * factor + 1)
    roi = big[r_lo:r_hi, c_lo:c_hi]
    py, px = np.unravel_index(np.argmax(roi), roi.shape)
    py += r_lo
    px += c_lo
    range_cut = big[py, :]    # 距离向（列）
    az_cut = big[:, px]       # 方位向（行）

    out = {}
    for name, cut in (("range", range_cut), ("azimuth", az_cut)):
        w3 = _half_power_width(cut, factor)
        idx = int(np.argmax(cut))
        l, r = _mainlobe_bounds(cut, idx)
        out[name] = {
            "resolution_px": float(w3),
            "pslr_db": _pslr(cut, l, r),
            "islr_db": _islr(cut, l, r),
            "sslr_db": _sslr(cut, l, r),
        }
    out["peak_db"] = float(to_db(np.asarray([[d[y, x]]]))[0, 0])
    out["window"] = [y0, y1, x0, x1]
    out["_cuts"] = {"range": range_cut, "azimuth": az_cut, "factor": factor}
    return out


def _pslr(cut: np.ndarray, l: int, r: int) -> float | None:
    peak = float(cut.max())
    if peak <= 0:
        return None
    s = cut.copy()
    s[l:r + 1] = 0.0
    sl = float(s.max())
    if sl <= 0:
        return None
    return 20.0 * np.log10(sl / peak)


def _islr(cut: np.ndarray, l: int, r: int) -> float | None:
    main_e = float((cut[l:r + 1] ** 2).sum())
    total_e = float((cut ** 2).sum())
    side_e = total_e - main_e
    if main_e <= 0 or side_e <= 0:
        return None
    return 10.0 * np.log10(side_e / main_e)


def _sslr(cut: np.ndarray, l: int, r: int) -> float | None:
    """次生旁瓣比：排除主瓣及紧邻旁瓣（±1 主瓣跨度）后的最高旁瓣。"""
    peak = float(cut.max())
    if peak <= 0:
        return None
    idx = int(np.argmax(cut))
    span = r - l
    lo = max(0, idx - 2 * span)
    hi = min(len(cut), idx + 2 * span + 1)
    s = cut.copy()
    s[lo:hi] = 0.0
    far = float(s.max())
    if far <= 0:
        return None
    return 20.0 * np.log10(far / peak)
