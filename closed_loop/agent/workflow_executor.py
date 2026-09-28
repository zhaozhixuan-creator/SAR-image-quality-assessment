"""闭环执行器：评估 → 弱点检测 → 反馈（重选/调参/重训）→ 复评验证 → 输出画像。

默认运行在「复用模式」：读取 v3 既有评估结果（results/metrics/*/*.json），在纯逻辑层
完成反馈与验证，不消耗 GPU。全量生成模式由 scripts/run_feedback_experiments.py 编排。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.feedback_optimizer import FeedbackOptimizer
from evaluation.metrics_bridge import load_existing_metrics
from models.registry import ModelRegistry
from tools.io_utils import ensure_dir, write_json


class WorkflowExecutor:
    def __init__(self, system_config_path: str = "configs/system.yaml") -> None:
        from tools.io_utils import load_config
        self.system_config = load_config(system_config_path)
        self.paths = self.system_config["paths"]
        self.model_registry = ModelRegistry()
        self.feedback_optimizer = FeedbackOptimizer(system_config_path)

    def execute(self, task_schema: dict[str, Any], route: dict[str, Any],
                plan: dict[str, Any]) -> dict[str, Any]:
        model_id = route["model_id"]
        variant = route["variant"]
        quality_reqs = task_schema.get("quality_requirements", {})

        # 1. 复用 v3 既有指标（该模型所有变体）
        existing = load_existing_metrics()
        candidates = {
            v: existing[f"{model_id}/{v}"]
            for v in self.model_registry.get_metadata(model_id).get("variants", [])
            if f"{model_id}/{v}" in existing
        }
        current = candidates.get(variant)

        if current is None:
            return self._blocked(task_schema, route, plan,
                                 f"模型 {model_id}/{variant} 无既有评估结果，需先运行 v3 流水线 generate/evaluate")

        # 2. 反馈（弱点检测 → 动作）
        v3_model_cfg = self.model_registry.v3_model_cfg(model_id)
        feedback = self.feedback_optimizer.suggest(
            route, current, quality_reqs, candidates, v3_model_cfg)

        # 3. 复评验证：若重选，改派变体并计算指标改善
        improvement = None
        final_variant = variant
        final_metrics = current
        if feedback.get("reselect", {}).get("applied"):
            new_variant = feedback["reselect"]["to_variant"]
            new_metrics = candidates.get(new_variant)
            if new_metrics is not None:
                exp = feedback["reselect"]["expected"]
                metric = exp["metric"]
                improvement = {
                    "metric": metric,
                    "from_variant": variant,
                    "to_variant": new_variant,
                    "from_value": exp["from_value"],
                    "to_value": exp["to_value"],
                    "relative_gap": exp["relative_gap"],
                }
                final_variant = new_variant
                final_metrics = new_metrics

        # 4. 输出结构化画像 portrait.json
        portrait = self._build_portrait(task_schema, route, plan, model_id,
                                        final_variant, final_metrics, feedback, improvement)
        record_dir = ensure_dir(Path(self.paths["task_record_dir"]))
        record_path = write_json(record_dir / f"{task_schema['task_id']}.json", portrait)

        return {
            "task_id": task_schema["task_id"],
            "status": "completed",
            "selected_model": model_id,
            "selected_variant": final_variant,
            "selection_reason": feedback.get("reselect", {}).get("reason", "") if improvement else route.get("reason", ""),
            "verdict": feedback["verdict"],
            "feedback": feedback,
            "improvement": improvement,
            "portrait": portrait,
            "task_record": str(record_path),
        }

    @staticmethod
    def _build_portrait(task_schema, route, plan, model_id, variant, metrics,
                        feedback, improvement) -> dict[str, Any]:
        return {
            "request": task_schema,
            "route": {k: v for k, v in route.items()},
            "plan": plan,
            "selected_model": model_id,
            "selected_variant": variant,
            "metrics": {k: metrics.get(k) for k in
                        ["fid", "ssim", "afs", "delta_enl", "bve", "cmae_deg",
                         "mse", "rmse", "psnr", "ncc", "uqi", "ms_ssim", "fsim"]},
            "characterization": feedback["portrait"]["characterization"],
            "verdict": feedback["verdict"],
            "feedback": {
                "action": feedback["action"],
                "reselect": feedback.get("reselect"),
                "retrain": feedback.get("retrain"),
                "main_issues": feedback["main_issues"],
                "recommended_actions": feedback["recommended_actions"],
            },
            "improvement": improvement,
        }

    @staticmethod
    def _blocked(task_schema, route, plan, message: str) -> dict[str, Any]:
        return {
            "task_id": task_schema["task_id"],
            "status": "blocked",
            "message": message,
            "task_schema": task_schema,
            "route": route,
            "plan": plan,
        }
