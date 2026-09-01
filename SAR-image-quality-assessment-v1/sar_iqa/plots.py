"""matplotlib 图表生成：输出 base64 PNG，供 HTML 报告内嵌。"""
from __future__ import annotations

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .profiles import _c0, to_db

# 中文字体兜底
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def thumbnail(sar, max_side: int = 480) -> str:
    """图像缩略图（dB 对数显示）。"""
    db = to_db(_c0(sar.intensity))
    vmin, vmax = np.percentile(db, 2), np.percentile(db, 98)
    if vmax <= vmin:
        vmax = vmin + 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(db, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"输入图像（dB 显示，{sar.W}×{sar.H}）")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
    return _b64(fig)


def profiles(row_prof: np.ndarray, col_prof: np.ndarray) -> str:
    """行/列均值剖面。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(row_prof, lw=0.8)
    axes[0].set_title("方位向行均值剖面（扇贝 / 丢行）")
    axes[0].set_xlabel("行"); axes[0].set_ylabel("dB")
    axes[1].plot(col_prof, lw=0.8, color="tab:orange")
    axes[1].set_title("距离向列均值剖面（EAP / 子带台阶）")
    axes[1].set_xlabel("列"); axes[1].set_ylabel("dB")
    fig.tight_layout()
    return _b64(fig)


def histogram(intensity: np.ndarray) -> str:
    """直方图（dB）—— 观察饱和堆积。"""
    d = to_db(_c0(intensity))
    d = d[np.isfinite(d)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(d, bins=100, color="steelblue", alpha=0.85)
    ax.set_title("直方图（dB）")
    ax.set_xlabel("dB"); ax.set_ylabel("像元数")
    fig.tight_layout()
    return _b64(fig)


def irf(irf: dict) -> str:
    """点目标 IRF：过采样窗口 + 距离/方位剖线。"""
    cuts = irf["_cuts"]
    range_cut = cuts["range"]
    az_cut = cuts["azimuth"]
    factor = cuts["factor"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    r_db = 20 * np.log10(np.maximum(range_cut, 1e-9))
    a_db = 20 * np.log10(np.maximum(az_cut, 1e-9))
    xr = np.arange(len(range_cut)) / factor
    xa = np.arange(len(az_cut)) / factor
    axes[0].plot(xr, r_db, lw=0.7)
    axes[0].set_title("距离向剖线（幅度 dB）"); axes[0].set_xlabel("像素")
    axes[1].plot(xa, a_db, lw=0.7, color="tab:red")
    axes[1].set_title("方位向剖线（幅度 dB）"); axes[1].set_xlabel("像素")
    for ax in (axes[0], axes[1]):
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _b64(fig)
