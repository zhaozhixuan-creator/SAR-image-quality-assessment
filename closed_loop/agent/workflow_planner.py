"""工作流规划：闭环 7 步（解析→路由→评估→弱点→反馈→验证→画像）。"""
from __future__ import annotations

from typing import Any


class WorkflowPlanner:
    def plan(self, task_schema: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": task_schema["task_id"],
            "steps": [
                {"name": "parse_requirement", "status": "ready"},
                {"name": "route_model", "status": "ready",
                 "model_id": route["model_id"], "variant": route["variant"]},
                {"name": "evaluate_quality", "status": "ready"},
                {"name": "detect_weakness", "status": "ready"},
                {"name": "apply_feedback", "status": "ready"},      # reselect / adjust_param / retrain
                {"name": "verify_improvement", "status": "ready"},  # 复评 → 指标改善
                {"name": "output_portrait", "status": "ready"},     # portrait.json
            ],
        }
