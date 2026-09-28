#!/usr/bin/env python3
"""exp4 —— 方位角定向重训（质检估计器 R 作为冻结教师）。

与 exp3（朴素续训 200 kimg）**同预算**对比：同样 200 kimg 续训，但 Gmain 额外增加
「方位角一致性损失」——质检系统的辅助方位角估计器 R(·)（即计算 CMAE 的那个网络，
论文 §3.1.4）作为冻结教师，约束生成图朝向 = 条件方位角 (sin,cos)。

这是「质检定位弱点(CMAE≈91°，近似随机)→ 定向修复(方位角损失微调)→ 指标改善」的闭环落地，
对比 exp3 朴素续训无改善，证明质检的反馈动作（用 R 定向微调）确有因果效用。

用法：
  python scripts/run_azimuth_finetune.py --gpu 2 --kimg 200 --az-weight 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.io_utils import ensure_dir, load_config, write_json
from evaluation.metrics_bridge import load_existing_metrics, v3_root, load_v3_config
from evaluation.portrait import ALL_KEYS
from scripts.run_feedback_experiments import _stylegan_env, _evaluate_stylegan

RESULT_DIR = ensure_dir(PROJECT_ROOT / "results" / "experiments")


def main() -> None:
    cli = argparse.ArgumentParser(description="exp4 方位角定向重训")
    cli.add_argument("--gpu", type=int, default=2)
    cli.add_argument("--kimg", type=int, default=200)
    cli.add_argument("--az-weight", type=float, default=1.0)
    cli.add_argument("--per-class", type=int, default=100)
    cli.add_argument("--batch", type=int, default=32)
    args = cli.parse_args()

    cfg = load_v3_config()
    m = cfg["models"]["stylegan4sar"]
    model_root, env = _stylegan_env(cfg, args.gpu)
    venv_py = m["venv_python"].replace("{root}", cfg["root"])
    variant = m["variants"]["stylegan2_ada_5class"]
    resume_pkl = variant["checkpoint"].replace("{root}", cfg["root"])
    data_zip = f"{model_root}/datasets/mstar15_5class_stride5_maxdev-128.zip"
    r_ckpt = str(v3_root() / "workspace" / "checkpoints" / "R_128.pt")
    outdir = str(ensure_dir(PROJECT_ROOT / "results" / "retrain" / "stylegan_azimuth"))

    print(f"[exp4] 方位角定向续训 --resume={resume_pkl} --kimg={args.kimg} "
          f"--azimuth_weight={args.az_weight} …", flush=True)
    subprocess.run([
        venv_py, "train.py", f"--outdir={outdir}", f"--data={data_zip}",
        "--gpus=1", "--cond=1", "--cfg=auto", "--aug=ada", f"--batch={args.batch}",
        f"--kimg={args.kimg}", "--snap=25", "--seed=0", "--metrics=none", "--workers=3",
        f"--resume={resume_pkl}",
        f"--azimuth_weight={args.az_weight}", f"--azimuth_estimator={r_ckpt}",
    ], cwd=model_root, env=env, check=True)

    snapshots = sorted(Path(outdir).rglob("network-snapshot-*.pkl"),
                       key=lambda p: int(p.stem.split("-")[-1]))
    if not snapshots:
        raise RuntimeError("续训未产出 network-snapshot-*.pkl")
    new_pkl = snapshots[-1]
    print(f"[exp4] 用新 snapshot 生成：{new_pkl} …", flush=True)

    gen_dir = ensure_dir(PROJECT_ROOT / "results" / "generated" / "stylegan_azimuth")
    angles_json = v3_root() / "workspace" / "real" / "128" / "train" / "labels_stylegan.json"
    if not angles_json.exists():
        raise RuntimeError("缺少 labels_stylegan.json，请先运行 v3 --stage prepare/generate")
    classes = m.get("classes") or cfg["datasets"]["class_names"]
    subprocess.run([
        venv_py, str(v3_root() / "generators" / "stylegan_generate.py"),
        "--checkpoint", str(new_pkl), "--outdir", str(gen_dir),
        "--angles-json", str(angles_json), "--per-class", str(args.per_class),
        "--aasg-enabled", "0", "--classes", str(len(classes)),
        "--device", f"cuda:{args.gpu}",
    ], cwd=m["cwd"].replace("{root}", cfg["root"]), check=True)

    man_path = gen_dir / "generation_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for s in man["samples"]:
        s["class_name"] = classes[s["class_idx"]]
    write_json(man_path, man)

    print("[exp4] 评估 13 项指标 …", flush=True)
    new_metrics = _evaluate_stylegan(cfg, gen_dir, classes, args.gpu)

    baseline = load_existing_metrics().get("stylegan4sar/stylegan2_ada_5class", {})

    def delta(k):
        a, b = new_metrics.get(k), baseline.get(k)
        return (a - b) if (a is not None and b is not None) else None

    payload = {
        "experiment": "exp4_azimuth_retrain",
        "mechanism": "方位角定向重训（质检估计器 R 作为冻结教师，方位角一致性损失）",
        "question": "质检发现的弱点（CMAE≈91°，方位角控制≈随机）经「用 R 定向微调」后是否改善？",
        "resume_checkpoint": resume_pkl,
        "new_checkpoint": str(new_pkl),
        "kimg": args.kimg,
        "azimuth_weight": args.az_weight,
        "azimuth_estimator": r_ckpt,
        "n_generated": len(man["samples"]),
        "baseline_metrics": {k: baseline.get(k) for k in ALL_KEYS},
        "retrained_metrics": {k: new_metrics.get(k) for k in ALL_KEYS},
        "delta": {k: delta(k) for k in ALL_KEYS},
        "conclusion": "见 delta 列",
    }
    write_json(RESULT_DIR / "exp4_azimuth_retrain.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
