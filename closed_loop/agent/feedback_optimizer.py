"""反馈优化器：把「质检画像 + 弱点」转化为可执行的反馈动作（重选 / 调参 / 重训）。

这是闭环的决策点：质检不再只给分数，而是据此决定「下一步该动什么」。
"""
from __future__ import annotations

from typing import Any

from evaluation.portrait import build_portrait, suggest_action
from feedback.reselect import reselect
from feedback.retrain import build_retrain_action


class FeedbackOptimizer:
    def __init__(self, system_config_path: str = "configs/system.yaml") -> None:
        from tools.io_utils import load_config
        self.system_config = load_config(system_config_path)

    def suggest(self, route: dict[str, Any], current_metrics: dict[str, Any],
                quality_requirements: dict[str, str],
                candidates: dict[str, dict[str, Any]] | None = None,
                v3_model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """返回 feedback dict：{pass, portrait, action, reselect?, retrain?, main_issues, actions}。"""
        model_id = route["model_id"]
        variant = route["variant"]

        portrait = build_portrait(model_id, variant, current_metrics,
                                  request=None, candidates=list(candidates or {}))
        weak_metrics = set(portrait["characterization"]["weak_metrics"])
        verdict = portrait["verdict"]

        issues = portrait["characterization"]["weaknesses"]
        actions: list[str] = []
        reselect_result = None
        retrain_result = None

        if verdict == "pass":
            actions.append("质检通过，无需反馈动作")
            return {
                "pass": True,
                "verdict": verdict,
                "portrait": portrait,
                "action": "none",
                "main_issues": [],
                "recommended_actions": actions,
            }

        # 决策：背景统计类弱点 → 优先重选；否则重训
        if weak_metrics and candidates and (weak_metrics & {"delta_enl", "bve"}):
            reselect_result = reselect(model_id, variant, quality_requirements, candidates)
            if reselect_result.get("applied"):
                actions.append(f"重选模型变体：{variant} → {reselect_result['to_variant']}"
                               f"（{reselect_result['reason']}）")
            else:
                reselect_result = None

        if not reselect_result and v3_model_cfg is not None:
            retrain_result = build_retrain_action(model_id, weak_metrics, v3_model_cfg, variant)
            actions.append(f"触发重训：{retrain_result.get('reason')}"
                           f"（{retrain_result.get('method')}）")

        return {
            "pass": False,
            "verdict": verdict,
            "portrait": portrait,
            "action": "reselect" if reselect_result else ("retrain" if retrain_result else "none"),
            "reselect": reselect_result,
            "retrain": retrain_result,
            "main_issues": issues,
            "recommended_actions": actions,
        }
