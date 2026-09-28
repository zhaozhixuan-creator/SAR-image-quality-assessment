"""资源路由：task_schema → 模型/变体（选型速查表的代码化）。

规则：
  1. 按 task_type 匹配唯一模型（三任务三模型，不重叠）。
  2. 校验类别边界：batch_generation 仅 5 类；越界 → 拒绝/降级为 novel_view_synthesis。
  3. 初始变体 = 模型默认变体（反馈阶段可重选）。
"""
from __future__ import annotations

from typing import Any

from tools.io_utils import load_config


class ResourceRouter:
    def __init__(self, models_config_path: str = "configs/models.yaml") -> None:
        self.models_config = load_config(models_config_path)
        self.models = {m["model_id"]: m for m in self.models_config["models"]}

    def route(self, task_schema: dict[str, Any]) -> dict[str, Any]:
        task_type = task_schema["task_type"]
        model = self._model_for_task(task_type)
        variant = model["default_variant"]

        # 类别边界校验
        classes = task_schema.get("target_classes", [])
        supported = model.get("supported_classes", [])
        unsupported = [c for c in classes if supported and c not in supported]
        blocked = model["task_type"] == "batch_generation" and bool(unsupported)

        reason = self._reason(task_type, model, blocked, unsupported)

        return {
            "model_id": model["model_id"],
            "variant": variant,
            "task_type": task_type,
            "requires_reference": model.get("requires_reference", False),
            "supported_classes": supported,
            "size": model.get("size"),
            "applicable_metrics": model.get("applicable_metrics", []),
            "blocked": blocked,
            "unsupported_classes": unsupported,
            "reason": reason,
        }

    def _model_for_task(self, task_type: str) -> dict[str, Any]:
        for model in self.models_config["models"]:
            if model["task_type"] == task_type:
                return model
        raise KeyError(f"无模型覆盖任务类型：{task_type}")

    @staticmethod
    def _reason(task_type: str, model: dict[str, Any], blocked: bool, unsupported: list[str]) -> str:
        if blocked:
            return (f"{model['model_name']}仅支持 {model['supported_classes']}，"
                    f"请求含越界类别 {unsupported} → 无参考批量生成不可执行")
        if task_type == "batch_generation":
            return "无参考、类+方位角批量生成 → stylegan4sar（限 5 类）"
        if task_type == "novel_view_synthesis":
            return "参考图驱动换视角 → angle_gen（10 类、保实例）"
        return "X→Ka 跨波段翻译 → frequency_gen（仅此域对）"
