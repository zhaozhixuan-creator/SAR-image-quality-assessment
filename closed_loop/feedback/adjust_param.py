"""反馈机制 2 —— 生成参数调整。

在单模型内扫一个生成超参数（如扩散去噪步数 / truncation ψ），对每个取值评估质检指标，
分析「指标是否随参数单调变化」以证明质检对参数敏感（可指导调参），并选出最优取值。

本模块只做分析逻辑；参数扫描的生成/评估由 scripts/run_feedback_experiments.py 编排。
"""
from __future__ import annotations

from typing import Any

from evaluation.portrait import IDEAL


def analyze_sweep(param_values: list[float], metrics: dict[float, dict[str, Any]],
                  metric: str) -> dict[str, Any]:
    """给定 {参数取值: 指标 dict}，返回质检敏感性与最优取值。

    返回：
      - monotonic: 质量是否随参数单调改善（越大越好指标→单调增；越小越好→单调减）
      - best_param: 该指标方向上的最优参数取值
      - improvement: 相对首尾/最差最优的相对改善
      - series: 参数→指标值（供画曲线）
    """
    ideal = IDEAL[metric]
    series = sorted((p, m.get(metric)) for p, m in metrics.items()
                    if m.get(metric) is not None)
    if len(series) < 2:
        return {"monotonic": False, "reason": "有效参数点不足 2 个，无法判断敏感性", "series": series}

    vals = [v for _, v in series]
    params = [p for _, p in series]

    # 质检敏感性：指标随参数单调变化（无论增/减），即证明质检对参数可预测地响应。
    increasing = all(b >= a for a, b in zip(vals, vals[1:]))
    decreasing = all(b <= a for a, b in zip(vals, vals[1:]))
    monotonic = increasing or decreasing
    direction = "increasing" if increasing else ("decreasing" if decreasing else "none")

    best_val = min(vals) if ideal < 0 else max(vals)
    worst_val = max(vals) if ideal < 0 else min(vals)
    best_param = params[vals.index(best_val)]
    ref = max(abs(best_val), abs(worst_val), 1e-9)
    improvement = abs(best_val - worst_val) / ref

    return {
        "metric": metric,
        "monotonic": monotonic,
        "direction": direction,
        "best_param": best_param,
        "best_value": best_val,
        "improvement": round(improvement, 4),
        "series": [{"param": p, "value": v} for p, v in series],
    }
