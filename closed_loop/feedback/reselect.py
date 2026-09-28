"""反馈机制 1 —— 模型/变体重选。

给定需求的质量侧重点 + 同模型各变体的实测指标，检测当前变体是否在关注维度上偏弱，
若是则改派该维度上指标最优的变体，并给出预期改进幅度。

质量需求 → 指标映射：
  background_consistency → delta_enl / bve（越小越好）
  structure_fidelity      → ssim / afs（越大越好）
  azimuth_precision       → cmae_deg（越小越好）
"""
from __future__ import annotations

from typing import Any

from evaluation.portrait import IDEAL

# quality_requirement 字段 → 关心的指标（越小/越大由 IDEAL 决定）
_REQ_METRICS = {
    "background_consistency": ["delta_enl", "bve"],
    "structure_fidelity": ["ssim", "afs"],
    "azimuth_precision": ["cmae_deg"],
}


def _best_variant(metrics_by_variant: dict[str, dict[str, Any]], metric: str) -> str | None:
    """在变体间挑 metric 方向最优者（缺失该指标则跳过该变体）。"""
    ideal = IDEAL[metric]
    best_variant, best_val = None, None
    for variant, m in metrics_by_variant.items():
        v = m.get(metric)
        if v is None:
            continue
        better = (best_val is None) or (v < best_val if ideal < 0 else v > best_val)
        if better:
            best_variant, best_val = variant, v
    return best_variant


def reselect(model_id: str, current_variant: str, quality_requirements: dict[str, str],
             metrics_by_variant: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """返回 {action, reselect, from_variant, to_variant, reason, expected}。

    metrics_by_variant: {variant: 完整 13 项指标 dict}（至少含当前变体）。
    """
    # 只关注需求为 high 的维度
    focus = [f for f, level in quality_requirements.items() if level == "high"]
    metrics_of_interest = [m for f in focus for m in _REQ_METRICS.get(f, [])]

    if not metrics_of_interest or current_variant not in metrics_by_variant:
        return {"action": "reselect", "applied": False,
                "reason": "无关注维度或无当前变体指标，无法重选"}

    current = metrics_by_variant[current_variant]
    recommendations = []
    for metric in metrics_of_interest:
        best_v = _best_variant(metrics_by_variant, metric)
        cur_val = current.get(metric)
        if best_v is None or cur_val is None or best_v == current_variant:
            continue
        best_val = metrics_by_variant[best_v][metric]
        ideal = IDEAL[metric]
        # 当前变体在关注维度上确实更差（且差距 ≥5% 视为显著）
        worse = cur_val > best_val if ideal < 0 else cur_val < best_val
        gap = abs(cur_val - best_val) / max(abs(cur_val), abs(best_val), 1e-9)
        if worse and gap >= 0.05:
            recommendations.append({
                "metric": metric,
                "from_variant": current_variant, "from_value": cur_val,
                "to_variant": best_v, "to_value": best_val,
                "relative_gap": round(gap, 4),
            })

    if not recommendations:
        return {"action": "reselect", "applied": False,
                "reason": "当前变体在关注维度上非显著偏弱，无需重选"}

    # 按关注维度优先级取首条（或合并为「改派到综合更优变体」）
    top = recommendations[0]
    return {
        "action": "reselect",
        "applied": True,
        "model_id": model_id,
        "from_variant": current_variant,
        "to_variant": top["to_variant"],
        "reason": (f"需求关注 {focus}，当前 {current_variant} 的 {top['metric']}="
                   f"{top['from_value']:.4g} 弱于 {top['to_variant']} "
                   f"({top['to_value']:.4g})，改派该变体"),
        "expected": {
            "metric": top["metric"],
            "from_value": top["from_value"],
            "to_value": top["to_value"],
            "relative_gap": top["relative_gap"],
        },
        "recommendations": recommendations,
    }
