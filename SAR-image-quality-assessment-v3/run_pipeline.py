#!/usr/bin/env python3
"""SAR 生成图像质检评估工作流 —— 顶层编排器。

用法：
    python run_pipeline.py                          # 按 config.pipeline.stages 顺序全量执行
    python run_pipeline.py --stage prepare          # 只跑数据准备
    python run_pipeline.py --stage generate --model stylegan4sar
    python run_pipeline.py --stage evaluate --device cpu

各阶段见 pipeline/stage_*.py。阶段之间只通过 workspace/results 下的产物解耦。
"""
from __future__ import annotations

import argparse
import sys

from common import paths
from pipeline.stage_prepare import run as run_prepare
from pipeline.stage_generate import run as run_generate
from pipeline.stage_align import run as run_align
from pipeline.stage_evaluate import run as run_evaluate
from pipeline.stage_report import run as run_report

# 阶段注册表：显式声明而非 f"pipeline.stage_{name}" 字符串拼接。
# 每个 stage 是一个独立模块（run(cfg, args)），未来可 1:1 替换为独立 agent/subprocess。
STAGES = {
    "prepare": run_prepare,
    "generate": run_generate,
    "align": run_align,
    "evaluate": run_evaluate,
    "report": run_report,
}


def parse_args():
    ap = argparse.ArgumentParser(description="v3 SAR 生成图像质检评估工作流")
    ap.add_argument("--config", default=None, help="config.yaml 路径（默认 v3 根下）")
    ap.add_argument("--stage", default=None,
                    help="只执行指定阶段：prepare/generate/align/evaluate/report")
    ap.add_argument("--model", default=None, help="仅处理指定模型（generate/evaluate 用）")
    ap.add_argument("--device", default=None, help="覆盖评估设备：cpu/cuda/cuda:2")
    ap.add_argument("--gpu", type=int, default=None, help="覆盖评估 GPU 编号")
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(paths.V3_ROOT))
    cfg = paths.load_config(args.config)

    if args.device is not None:
        cfg.setdefault("evaluation", {})["device"] = args.device
    if args.gpu is not None:
        cfg.setdefault("evaluation", {})["gpu"] = args.gpu

    stages = [args.stage] if args.stage else cfg["pipeline"]["stages"]
    for name in stages:
        if name not in STAGES:
            print(f"[run_pipeline] 未知阶段：{name}（可选：{', '.join(STAGES)}）")
            raise SystemExit(1)
        print(f"\n===== Stage {name} =====")
        STAGES[name](cfg, args)


if __name__ == "__main__":
    main()
