#!/usr/bin/env python3
"""补齐 exp3 的最后一步：对已完成续训+生成的 stylegan_finetuned 做 13 指标评估并汇总。

续训（train.py --resume，kimg=200）与生成（500 张）已由 run_feedback_experiments.py --exp 3
完成；本脚本只做评估 + 对比基线 + 写 exp3_retrain.json，避免重跑 1 小时训练。
在 v3 评估环境（angle_gen/.venv）内运行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.io_utils import ensure_dir, write_json
from evaluation.metrics_bridge import load_existing_metrics, load_v3_config
from evaluation.portrait import ALL_KEYS
from scripts.run_feedback_experiments import _evaluate_stylegan

RESULT_DIR = ensure_dir(PROJECT_ROOT / "results" / "experiments")
GEN_DIR = PROJECT_ROOT / "results" / "generated" / "stylegan_finetuned"
SNAP = "results/retrain/stylegan/00000-mstar15_5class_stride5_maxdev-128-cond-auto1-kimg200-batch32-ada-resumecustom/network-snapshot-000200.pkl"


def main() -> None:
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument("--gpu", type=int, default=3)
    args = cli.parse_args()

    cfg = load_v3_config()
    m = cfg["models"]["stylegan4sar"]
    classes = m.get("classes") or cfg["datasets"]["class_names"]
    resume_pkl = m["variants"]["stylegan2_ada_5class"]["checkpoint"]

    print("[finish-exp3] 评估 500 张 fine-tuned 生成图 …", flush=True)
    new_metrics = _evaluate_stylegan(cfg, GEN_DIR, classes, args.gpu)

    baseline = load_existing_metrics().get("stylegan4sar/stylegan2_ada_5class", {})

    def delta(k):
        a, b = new_metrics.get(k), baseline.get(k)
        return (a - b) if (a is not None and b is not None) else None

    payload = {
        "experiment": "exp3_retrain",
        "mechanism": "重训（stylegan2-ada resume 续训 kimg=200）",
        "question": "质检发现的弱点（结构保真弱 SSIM/AFS）经重训后，指标是否改善？",
        "resume_checkpoint": resume_pkl,
        "new_checkpoint": str((PROJECT_ROOT / SNAP).resolve()),
        "kimg": 200,
        "n_generated": 500,
        "baseline_metrics": {k: baseline.get(k) for k in ALL_KEYS},
        "retrained_metrics": {k: new_metrics.get(k) for k in ALL_KEYS},
        "delta": {k: delta(k) for k in ALL_KEYS},
        "conclusion": "见 delta 列",
    }
    write_json(RESULT_DIR / "exp3_retrain.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
