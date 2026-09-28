"""模型注册：闭环侧的能力元数据 + 指向 v3 生成器/配置的入口。

- `get_metadata(model_id)`：能力边界（configs/models.yaml）。
- `v3_config()` / `v3_model_cfg(model_id)`：v3 config.yaml 的生成参数（checkpoint/venv/cwd）。
- `generate_via_v3(model_id)`：全量模式下委托 v3 流水线生成（shell 到 run_pipeline.py）。

说明：三个生成模型的推理包装在 v3/generators/*_generate.py，此处不复制实现；
「重选模型」反馈复用 v3 既有评估结果，「调参/重训」实验由 scripts 自行编排定制生成。
"""
from __future__ import annotations

import subprocess
from typing import Any

from tools.io_utils import load_config, resolve_path


class ModelRegistry:
    def __init__(self, models_config_path: str = "configs/models.yaml") -> None:
        self.models_config = load_config(models_config_path)
        self.models = {m["model_id"]: m for m in self.models_config["models"]}

    def get_metadata(self, model_id: str) -> dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self.models[model_id]

    def list_available(self) -> list[dict[str, Any]]:
        return list(self.models.values())

    @staticmethod
    def v3_config() -> dict[str, Any]:
        from evaluation.metrics_bridge import load_v3_config
        return load_v3_config()

    def v3_model_cfg(self, model_id: str) -> dict[str, Any]:
        return self.v3_config()["models"][model_id]

    def generate_via_v3(self, model_id: str, gpu: int | None = None) -> None:
        """全量模式：委托 v3 流水线生成（复用其 venv/checkpoint/对齐逻辑）。"""
        from evaluation.metrics_bridge import v3_root
        cmd = ["python", "run_pipeline.py", "--stage", "generate", "--model", model_id]
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        print(f"[registry] 委托 v3 生成：{' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(v3_root()), check=True)
