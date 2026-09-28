"""v3 质检引擎桥接：读取既有评估结果 / 委托 v3 计算 13 项指标。

- `load_existing_metrics()`：纯 JSON 读 v3/results/metrics/*/*.json（无 torch 依赖），
  供「重选模型」反馈直接复用已算好的指标。
- `compute_metrics()`：委托 v3 的 evaluation.metrics_bridge.compute_all（需在 v3 评估
  环境 angle_gen/.venv 内运行），供「调参 / 重训」实验对新增生成结果计算指标。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tools.io_utils import resolve_path

V3_REL = "../SAR-image-quality-assessment-v3"


def v3_root() -> Path:
    return resolve_path(V3_REL)


def load_v3_config() -> dict[str, Any]:
    import yaml
    cfg_path = resolve_path("../SAR-image-quality-assessment-v3/config.yaml")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_existing_metrics() -> dict[str, dict[str, Any]]:
    """返回 {f"{model}/{variant}": metric_dict}，来自 v3/results/metrics/。"""
    root = v3_root() / "results" / "metrics"
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir():
            continue
        for f in sorted(mdir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            key = f"{data.get('model')}/{data.get('variant')}"
            out[key] = data
    return out


def _load_v3_module(name: str, relpath: str):
    """按文件路径加载 v3 模块，避免与 closed_loop 同名包（evaluation）命名冲突。"""
    import importlib.util

    if str(v3_root()) not in sys.path:
        sys.path.insert(0, str(v3_root()))
    spec = importlib.util.spec_from_file_location(name, str(v3_root() / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_metrics(real, fake, angles_deg, R, E_asc, size: int = 64,
                    device: str = "cpu", metrics=None) -> dict[str, Any]:
    """委托 v3 metrics_bridge 计算 13 项指标（需在 v3 评估环境运行）。"""
    mb = _load_v3_module("v3_metrics_bridge", "evaluation/metrics_bridge.py")
    gm = mb.load_gm(load_v3_config())
    return mb.compute_all(gm, real, fake, angles_deg, R, E_asc, size=size, device=device,
                          metrics=metrics)
