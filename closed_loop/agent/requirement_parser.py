"""需求解析：把自然语言生成需求 → 结构化生成规格（task_schema）。

task_type 三枚举与 v3 三模型一一对应：
  - batch_generation      → stylegan4sar   （无参考、类+方位角批量生成）
  - novel_view_synthesis  → angle_gen      （参考图驱动换视角）
  - cross_band_translation → frequency_gen （X→Ka 跨波段翻译）

解析规则为确定性正则（不依赖 LLM，保证可复现）；LLM 版需求解析可由
SAR-Generation 的 agent/user_intent_agent/intent_parser.py 提供，此处保留确定性版本。
"""
from __future__ import annotations

import re
from typing import Any

from tools.io_utils import load_config, new_task_id


# 任务类型关键词（顺序即优先级）
_TASK_KEYWORDS = [
    ("cross_band_translation", ["跨波段", "x→ka", "x -> ka", "x2ka", "波段翻译", "跨频", "翻译成 ka", "翻译成ka"]),
    ("novel_view_synthesis", ["换视角", "新视角", "参考图", "视角合成", "改视角", "转视角", "目标视角"]),
    ("batch_generation", ["批量生成", "生成", "造图", "合成图", "无参考"]),
]

# 质量需求关键词 → quality_requirements 字段
_QUALITY_KEYWORDS = {
    "background_consistency": ["背景", "噪声统计", "enl", "斑点"],
    "structure_fidelity": ["结构", "保真", "散射结构", "形状"],
    "azimuth_precision": ["方位角", "角度", "朝向"],
}


class RequirementParser:
    def __init__(self, system_config_path: str = "configs/system.yaml",
                 dataset_config_path: str = "configs/datasets.yaml") -> None:
        self.system_config = load_config(system_config_path)
        self.dataset_config = load_config(dataset_config_path)

    def parse(self, request: str) -> dict[str, Any]:
        defaults = self.system_config["system"]
        dataset_name = self._extract_dataset(request) or defaults["default_dataset"]
        classes = self._extract_classes(request, dataset_name)
        task_type = self._extract_task_type(request, defaults["default_task_type"])

        return {
            "task_id": new_task_id(),
            "raw_request": request,
            "task_type": task_type,
            "dataset": dataset_name,
            "target_classes": classes,
            "generation_count": self._extract_count(request) or defaults.get("default_generation_count", 100),
            "image_size": [128, 128],
            "azimuth_requirement": "balanced" if ("方位角" in request or "角度" in request) else "not_specified",
            "reference": "reference" if task_type == "novel_view_synthesis" else None,
            "quality_requirements": self._extract_quality(request),
            "threshold_mode": defaults.get("default_threshold_mode", "adaptive"),
            "max_iteration": defaults["max_iteration"],
            "seed": 42,
        }

    # ---- 私有 ----
    def _extract_task_type(self, request: str, default: str) -> str:
        low = request.lower()
        for task_type, keywords in _TASK_KEYWORDS:
            if any(kw in low for kw in keywords):
                return task_type
        return default

    def _extract_dataset(self, request: str) -> str | None:
        for dataset in self.dataset_config.get("datasets", []):
            for name in [dataset["dataset_name"], *dataset.get("dataset_aliases", [])]:
                if name.lower() in request.lower():
                    return dataset["dataset_name"]
        return None

    def _extract_classes(self, request: str, dataset_name: str) -> list[str]:
        for dataset in self.dataset_config.get("datasets", []):
            if dataset["dataset_name"] != dataset_name:
                continue
            matches = [name for name in dataset.get("classes", []) if name.lower() in request.lower()]
            if matches:
                return matches
            alias_matches = [
                canonical
                for alias, canonical in dataset.get("class_aliases", {}).items()
                if alias.lower() in request.lower()
            ]
            if alias_matches:
                return list(dict.fromkeys(alias_matches))
            classes = dataset.get("classes", [])
            return [classes[0]] if classes else []
        return []

    @staticmethod
    def _extract_count(request: str) -> int | None:
        for pattern in [r"(\d+)\s*(张|个|幅)", r"(\d+)\s*(samples?|images?)"]:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                return max(1, int(match.group(1)))
        return None

    @staticmethod
    def _extract_quality(request: str) -> dict[str, str]:
        out = {}
        for field, keywords in _QUALITY_KEYWORDS.items():
            out[field] = "high" if any(kw in request for kw in keywords) else "medium"
        return out
