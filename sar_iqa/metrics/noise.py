"""维度 4 · 噪声与干扰（6 项）。"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label, uniform_filter

from ..base import MetricResult, Status
from ..profiles import _c0, to_db
from ..modules.despeckling import estimate_enl
from ..modules.rfi import image_proxy

DIM = "噪声与干扰"


def compute(ctx) -> list[MetricResult]:
    return [
        _nesz(ctx),
        _enl(ctx),
        _ambiguity(ctx),
        _rfi(ctx),
        _negative_rate(ctx),
        _dropout(ctx),
    ]


def _nesz(ctx):
    d = _c0(ctx.sar.intensity)
    local = to_db(uniform_filter(d, size=15))
    floor = float(np.percentile(local, 1.0))
    return MetricResult(
        key="noise.nesz", name="NESZ 噪声等效散射系数", dimension=DIM,
        value=floor, unit="dB", status=Status.WARN,
        threshold="TSX −19~−26 dB；GF-3 设计 −20 / 在轨约 −22 dB",
        reason="代理值：取局部均值最低 1% 分位作为噪声底。无绝对定标，需低散射参考区（Doldrums）交叉极化剖面 + 定标常数才可得真 NESZ。",
    )


def _enl(ctx):
    est = estimate_enl(ctx.sar.intensity)
    if est["n_pixels"] < 16:
        return MetricResult(key="noise.enl", name="ENL / 辐射分辨率", dimension=DIM,
                            status=Status.NODATA, unit="视数",
                            reason="均匀区太少，无法可靠估计 ENL。",
                            threshold="GF-3 辐射分辨率 3 dB（精细模式约 1.5 dB）")
    m, lc = est["enl_moment"], est["enl_logcumulant"]
    status = Status.PASS if not est["diverged"] else Status.WARN
    note = "双估计量一致。" if not est["diverged"] else "矩估计与对数累积量估计分歧明显 → 所选区块不够均匀，须重新筛选。"
    return MetricResult(
        key="noise.enl", name="ENL / 辐射分辨率", dimension=DIM,
        value=m, unit="视数", status=status,
        threshold="GF-3 辐射分辨率 3 dB（精细模式约 1.5 dB）；ENL 须与 EPD-ROA 成对报告",
        reason=note,
        detail={"enl_logcumulant": lc, "n_uniform_pixels": est["n_pixels"]},
        raw={"mask": est["mask"]},
    )


def _ambiguity(ctx):
    return MetricResult(
        key="noise.ambiguity", name="AASR / RASR 模糊比", dimension=DIM,
        status=Status.NODATA, unit="dB",
        threshold="TSX 方位 ≈ −20 dB、距离 ≈ −25 dB",
        reason="需方位谱复制能量占比与鬼影识别（依赖原始回波方位谱 / PRF），单张已成像产品无法计算。",
    )


def _rfi(ctx):
    r = image_proxy(ctx.sar.intensity)
    ratio = r["ratio"] * 100.0
    if ratio < 0.5:
        status, note = Status.PASS, "图像域未检出明显方位向亮条纹。"
    else:
        status, note = Status.WARN, f"图像域检出 {r['affected_rows']} 行疑似 RFI 亮条纹。"
    return MetricResult(
        key="noise.rfi", name="RFI 污染比例", dimension=DIM,
        value=ratio, unit="%", status=status,
        threshold="场景可用性标记，非合格性判据（强地域依赖，须逐场景检测）",
        reason=note + "（图像域代理；真实谱域检测需原始回波，见模块 rfi.detect_rfi）",
        detail={"affected_rows": r["affected_rows"]},
        raw={"row_profile": r["row_profile"], "threshold": r["threshold"]},
    )


def _negative_rate(ctx):
    raw = ctx.sar.data
    neg = float((raw < 0).mean()) * 100.0
    if neg < 0.1:
        status, note = Status.PASS, "负值占比可忽略。"
    elif neg < 5.0:
        status, note = Status.WARN, "存在少量负值（可能是热噪声减除后的过减）。"
    else:
        status, note = Status.FAIL, "负值占比过高，噪声矢量疑似被高估。"
    return MetricResult(
        key="noise.negative_rate", name="去噪负值率", dimension=DIM,
        value=neg, unit="%", status=status,
        threshold="过高 = 噪声被高估；过低 = 噪声未被充分减除",
        reason=note + ("（输入为幅度/强度域时通常恒为 0，此指标仅对已做噪声减除的浮点数据有意义）"
                       if ctx.sar.domain != "db" else ""),
    )


def _dropout(ctx):
    d = _c0(ctx.sar.intensity)
    db = to_db(d)
    row = db.mean(axis=1)
    col = db.mean(axis=0)

    bad_rows = _bad_lines(row)
    bad_cols = _bad_lines(col)

    invalid = ~np.isfinite(d) | (d <= 0)
    invalid_frac = float(invalid.mean()) * 100.0
    if invalid.any():
        lbl, n = label(invalid)
        sizes = np.bincount(lbl.ravel())
        sizes = sizes[1:] if len(sizes) > 1 else np.array([0])
        largest = float(sizes.max()) / d.size * 100.0 if sizes.size else 0.0
    else:
        largest = 0.0

    n_bad = int(bad_rows.sum() + bad_cols.sum())
    if n_bad == 0 and largest < 0.1:
        status, note = Status.PASS, "未检出丢行 / 大块坏像元。"
    elif largest > 1.0:
        status, note = Status.FAIL, f"检出大块无效区（最大连通域占 {largest:.2f}%）。"
    else:
        status, note = Status.WARN, f"检出 {n_bad} 行/列异常或零星坏像元。"
    return MetricResult(
        key="noise.dropout", name="丢行 / 坏像元率", dimension=DIM,
        value=invalid_frac, unit="%", status=status,
        threshold="A（大面积）/ D（零星）",
        reason=note,
        detail={"bad_rows": int(bad_rows.sum()), "bad_cols": int(bad_cols.sum()),
                "largest_cc_pct": largest},
        raw={"row_profile": row, "col_profile": col, "bad_rows": bad_rows, "bad_cols": bad_cols},
    )


def _bad_lines(prof: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    med = float(np.median(prof))
    mad = float(np.median(np.abs(prof - med)))
    sd = 1.4826 * mad if mad > 0 else float(prof.std())
    if sd <= 0:
        return np.zeros_like(prof, dtype=bool)
    return np.abs(prof - med) > sigma * sd
