"""自研模块 1 · 去斑评价指标包（对应 docs/08）。

核心判据：理想去斑的比值图（含斑 / 去斑）应为纯斑点（指数分布、无空间结构）。
本模块纯 numpy / scipy 实现，可复用于任意去斑 / 降噪算法的横向评测。

指标：均匀区自动筛选、ENL 双估计量（矩估计 + 对数累积量）、比值图白噪声检验、
EPD-ROA 边缘保持度、M-index（近似）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, sobel
from scipy.special import polygamma

EPS = 1e-12


def uniform_mask(intensity: np.ndarray, win: int = 7, cv_quantile: float = 0.5) -> np.ndarray:
    """自适应均匀区筛选：取局部变异系数（CV）最低的 cv_quantile 分位作为均匀区。

    斑点 L 视的 CV = 1/√L，纹理/边缘区 CV 更高，故最低 CV 分位对应最均匀区域。
    仅保留有效像元（有限且 >0），排除零填充 / NaN。
    """
    d = _c0(intensity)
    valid = np.isfinite(d) & (d > 0)
    mean = uniform_filter(d, size=win)
    sq_mean = uniform_filter(d ** 2, size=win)
    var = np.maximum(sq_mean - mean ** 2, 0.0)
    cv = np.sqrt(var) / np.maximum(mean, 1e-6)
    cvv = cv[valid]
    if cvv.size < 16:
        return valid
    thr = np.percentile(cvv, cv_quantile * 100.0)
    # 排除亮目标（点目标 / 舰船）：其主瓣局部平坦（低 CV）但远亮于背景，
    # 仅靠 CV 无法剔除，需额外要求局部均值不超过全局中位数的数倍。
    gmed = float(np.median(d[valid]))
    not_bright = mean <= 3.0 * max(gmed, 1e-6)
    return valid & (mean > 0) & (cv <= thr) & not_bright


def enl_moment(I: np.ndarray, mask: np.ndarray | None = None) -> float:
    """矩估计：ENL = μ² / σ²（均匀区最优，对纹理敏感）。"""
    x = I[mask] if mask is not None else I.ravel()
    if x.size < 4:
        return float("nan")
    mu = float(x.mean())
    var = float(x.var())
    if var <= 0:
        return float("inf")
    return mu ** 2 / var


def enl_logcumulant(I: np.ndarray, mask: np.ndarray | None = None) -> float:
    """对数累积量估计：由 Var[log I] = ψ₁(L) 反解 L（对纹理更稳健）。

    注：ψ₁(L)=polygamma(1,L) 随 L 单调递减，故二分分支需据此定向
    （ψ₁(mid) > v 表示 mid 过小，应抬升下界 lo）。
    """
    x = I[mask] if mask is not None else I.ravel()
    if x.size < 4:
        return float("nan")
    t = np.log(np.maximum(x, EPS))
    v = float(t.var())
    if v <= 0:
        return float("inf")
    lo, hi = 1e-3, 1e4
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if polygamma(1, mid) > v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_enl(intensity: np.ndarray, win: int = 7, cv_quantile: float = 0.5) -> dict:
    """ENL 双估计量 + 分歧告警。"""
    mask = uniform_mask(intensity, win=win, cv_quantile=cv_quantile)
    n = int(mask.sum())
    d = _c0(intensity)
    m = enl_moment(d, mask)
    lc = enl_logcumulant(d, mask)
    diverged = False
    if n > 0 and np.isfinite(m) and np.isfinite(lc):
        diverged = abs(m - lc) / max(m, lc, 1e-6) > 0.5
    return {
        "mask": mask,
        "n_pixels": n,
        "enl_moment": m,
        "enl_logcumulant": lc,
        "diverged": diverged,
    }


def lee_filter(intensity: np.ndarray, win: int = 7) -> np.ndarray:
    """简化 Lee 滤波（作为内部参考去斑基线）。"""
    d = _c0(intensity)
    mean = uniform_filter(d, size=win)
    sq_mean = uniform_filter(d ** 2, size=win)
    var = np.maximum(sq_mean - mean ** 2, 0.0)
    k = var / (var + mean ** 2 + EPS)
    return mean + k * (d - mean)


def spatial_autocorr(R: np.ndarray, lag: int = 1) -> float:
    """滞后 1 自相关（水平方向）。≈1 表示残留空间结构，≈0 表示白噪声。"""
    R = R.astype(np.float64)
    if R.ndim == 3:
        R = R[:, :, 0]
    a = R[:, :-lag].ravel()
    b = R[:, lag:].ravel()
    if a.size < 8:
        return 0.0
    am = a.mean()
    bm = b.mean()
    num = ((a - am) * (b - bm)).mean()
    den = np.sqrt(((a - am) ** 2).mean() * ((b - bm) ** 2).mean()) + EPS
    return float(num / den)


def ratio_image_test(I_speckled: np.ndarray, I_denoised: np.ndarray) -> dict:
    """比值图白噪声检验：比值图应逼近纯斑点（指数分布、无空间结构）。"""
    R = I_speckled / np.maximum(I_denoised, EPS)
    return {
        "mean_bias": float(abs(R.mean() - 1.0)),
        "var_bias": float(abs(R.var() - 1.0)),
        "spatial_ac": spatial_autocorr(R),
        "ratio": R,
    }


def detect_edges(intensity: np.ndarray, pct: float = 97.0) -> np.ndarray:
    """用 Sobel 梯度幅值阈值检测边缘。"""
    d = _c0(intensity)
    gx = sobel(d, axis=1)
    gy = sobel(d, axis=0)
    mag = np.hypot(gx, gy)
    return mag > np.percentile(mag, pct)


def epd_roa(I_ref: np.ndarray, I_den: np.ndarray, edges: np.ndarray | None = None,
            win: int = 3, max_pts: int = 4000) -> dict:
    """EPD-ROA：边缘两侧比值（Ratio of Averages）保持度。

    对检测到的边缘，比较去斑前后边缘两侧对比度保留程度。≈1 表示边缘保持良好。
    """
    I_ref = _c0(I_ref)
    I_den = _c0(I_den)
    if edges is None:
        edges = detect_edges(I_ref)
    gy, gx = np.gradient(I_ref.astype(np.float64))
    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        return {"epd": 1.0, "n_edges": 0}
    # 抽样控制计算量
    if ys.size > max_pts:
        idx = np.random.RandomState(0).choice(ys.size, max_pts, replace=False)
        ys, xs = ys[idx], xs[idx]

    H, W = I_ref.shape
    hw = win // 2
    roa_ref, roa_den = [], []
    for y, x in zip(ys, xs):
        nx, ny = float(gx[y, x]), float(gy[y, x])
        n = np.hypot(nx, ny) + EPS
        nx, ny = nx / n, ny / n
        y0a = int(np.clip(y - hw * ny, 0, H - 1))
        x0a = int(np.clip(x - hw * nx, 0, W - 1))
        y0b = int(np.clip(y + hw * ny, 0, H - 1))
        x0b = int(np.clip(x + hw * nx, 0, W - 1))
        def side_mean(img, y0, x0):
            y1 = min(H, y0 + win)
            x1 = min(W, x0 + win)
            return float(img[y0:y1, x0:x1].mean())
        a_r = side_mean(I_ref, y0a, x0a)
        b_r = side_mean(I_ref, y0b, x0b)
        a_d = side_mean(I_den, y0a, x0a)
        b_d = side_mean(I_den, y0b, x0b)
        roa_r = max(a_r, b_r) / max(min(a_r, b_r), EPS)
        roa_d = max(a_d, b_d) / max(min(a_d, b_d), EPS)
        roa_ref.append(roa_r)
        roa_den.append(roa_d)
    roa_ref = np.asarray(roa_ref)
    roa_den = np.asarray(roa_den)
    ratio = roa_den / np.maximum(roa_ref, EPS)
    epd = float(np.clip(np.median(ratio), 0.0, 2.0))
    return {"epd": epd, "n_edges": int(ys.size)}


def m_index(I_ref: np.ndarray, I_den: np.ndarray) -> float:
    """M-index（近似）：比值图一阶统计偏差 + 残留结构，越小越好。

    注意：本实现是 Gomez/Ospina/Frery 2017 M-index 的工程近似（一阶偏差 + 空间自相关），
    并非 UNASSISTED 的精确实现，跨论文比较前需与权威实现对齐。
    """
    r = ratio_image_test(I_ref, I_den)
    return float(r["mean_bias"] + r["var_bias"] + abs(r["spatial_ac"]))


def _c0(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x[:, :, 0]
    return x
