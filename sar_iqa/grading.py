"""判定与缺陷分级：连续量 → A/B/C/D 缺陷类 → 百分制 → 优/良/合格/不合格。

对应方案第五节，把物理指标接入 GB/T 24356「两级检查一级验收」的法定框架：
1. 超差比例 → 缺陷类（A/B/C/D）
2. 百分制扣分 S₂ = 100 − (12·a₁ + 4·a₂ + a₃)/t
3. 抽样量表（批量 → 样本）
4. 判决项 vs 标记项分离（§五.4）

注意：本模块输出的是**单图筛查评分**，最终验收仍需角反射器 / 时序等完整流水线。
"""
from __future__ import annotations

from .base import MetricResult, Status
from . import spec

# 超差比例 → 缺陷类 / 扣分（GB/T 24356 口径）
DEFECT_TABLE = [
    (5.0, "A", 42),
    (3.75, "B", 12),
    (2.5, "C", 4),
    (1.67, "D", 1),
]


def classify_fraction(pct: float) -> str | None:
    """把超差比例（%）映射到缺陷类 A/B/C/D；低于门槛返回 None。"""
    for thr, cls, _ in DEFECT_TABLE:
        if pct > thr:
            return cls
    return None


# GB/T 24356-2023 抽样量表（批量上限 → 样本量）。
# 锚点取自 README §五.3：批量 1–20 → 样本 3、687–1000 → 样本 56，
# 抽样比由 15% 递减至 5.6%，≥ 1001 须分批。
# 中间行按抽样比单调递减近似；生产环境应以 GB/T 24356-2023 原文量表替换。
SAMPLING_TABLE = [
    (20, 3),
    (50, 7),
    (100, 13),
    (150, 19),
    (200, 25),
    (300, 33),
    (400, 40),
    (500, 46),
    (600, 51),
    (686, 55),
    (1000, 56),
]


def sample_size(batch_size: int) -> dict:
    """由批量大小查抽样量，返回 {sample, batched, ratio_pct, reason}。

    ≥ 1001 须分批检验（batched=True, sample=None）。
    """
    if batch_size is None:
        return {"sample": None, "batched": False, "ratio_pct": None,
                "reason": "未提供批量大小。"}
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError):
        return {"sample": None, "batched": False, "ratio_pct": None,
                "reason": "批量大小非法。"}
    if batch_size <= 0:
        return {"sample": None, "batched": False, "ratio_pct": None,
                "reason": "批量大小非法。"}
    if batch_size > 1000:
        return {"sample": None, "batched": True, "ratio_pct": None,
                "reason": "批量 ≥ 1001，须分批检验。"}
    sample = None
    for upper, n in SAMPLING_TABLE:
        if batch_size <= upper:
            sample = n
            break
    if sample is None:
        sample = SAMPLING_TABLE[-1][1]
    ratio = sample / batch_size * 100.0
    return {"sample": sample, "batched": False, "ratio_pct": round(ratio, 2),
            "reason": f"批量 {batch_size} → 抽 {sample} 件（抽样比 {ratio:.2f}%）。"}


def grade(results: list[MetricResult]) -> dict:
    """由指标结果计算缺陷分级与百分制评分，并分离判决项 / 标记项。

    返回 dict：defects（判决项缺陷）、markers（标记项可用性元数据）、
    score、level、counts、evaluated、nodata。
    """
    evaluated = [r for r in results if r.status != Status.NODATA]
    t = len(evaluated)

    defects: list[dict] = []
    markers: list[dict] = []
    n_a = n_b = n_c = n_d = 0

    for r in results:
        if r.status not in (Status.FAIL, Status.WARN):
            continue
        # 标记项：场景相关退化（RFI / 叠掩阴影 / 低相干区），只作可用性元数据，
        # 不计入缺陷与评分（§五.4）。
        if r.kind == spec.MARKER:
            markers.append({"name": r.name, "dimension": r.dimension,
                            "value": r.value, "unit": r.unit, "reason": r.reason,
                            "level": r.level})
            continue
        cls = None
        if r.unit == "%" and r.value is not None:
            cls = classify_fraction(r.value)
        if cls is None:
            # 非比例型缺陷：FAIL 记 B 类，WARN 记 D 类（轻 / 需关注）
            cls = "B" if r.status == Status.FAIL else "D"
        defects.append({"name": r.name, "dimension": r.dimension,
                        "class": cls, "value": r.value, "unit": r.unit,
                        "reason": r.reason})
        if cls == "A":
            n_a += 1
        elif cls == "B":
            n_b += 1
        elif cls == "C":
            n_c += 1
        elif cls == "D":
            n_d += 1

    # 百分制扣分 S₂ = 100 − (12·a₁ + 4·a₂ + 1·a₃)/t
    if n_a > 0:
        score = 0.0
        level = "不合格"
        note = "出现 A 类（严重）缺陷，样本即判整批不合格。"
    else:
        penalty = (12 * n_b + 4 * n_c + 1 * n_d)
        score = max(0.0, 100.0 - penalty / max(t, 1))
        if score >= 90:
            level = "优"
        elif score >= 75:
            level = "良"
        elif score >= 60:
            level = "合格"
        else:
            level = "不合格"
        note = ""

    return {
        "defects": defects,
        "markers": markers,
        "score": score,
        "level": level,
        "note": note,
        "counts": {"A": n_a, "B": n_b, "C": n_c, "D": n_d},
        "evaluated": t,
        "nodata": len(results) - t,
    }
