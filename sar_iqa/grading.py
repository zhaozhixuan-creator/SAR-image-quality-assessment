"""判定与缺陷分级：连续量 → A/B/C/D 缺陷类 → 百分制 → 优/良/合格/不合格。

对应方案第五节，把物理指标接入 GB/T 24356「两级检查一级验收」的法定框架。
注意：本模块输出的是**单图筛查评分**，最终验收仍需角反射器 / 时序等完整流水线。
"""
from __future__ import annotations

from .base import MetricResult, Status

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


def grade(results: list[MetricResult]) -> dict:
    """由指标结果计算缺陷分级与百分制评分。

    返回 dict：defects、score、level、counts。
    """
    evaluated = [r for r in results if r.status != Status.NODATA]
    t = len(evaluated)

    defects: list[dict] = []
    n_a = n_b = n_c = n_d = 0

    for r in results:
        if r.status not in (Status.FAIL, Status.WARN):
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
        "score": score,
        "level": level,
        "note": note,
        "counts": {"A": n_a, "B": n_b, "C": n_c, "D": n_d},
        "evaluated": t,
        "nodata": len(results) - t,
    }
