#!/usr/bin/env python3
"""StyleGAN 条件生成封装（在 stylegan4SAR/.venv 内运行，cwd=stylegan4SAR 目录）。

按「类别 + 方位角」条件批量生成 SAR 图像：
    c = [onehot(10), sin(az), cos(az)]  （az 为弧度，与数据集 prepare 脚本口径一致）

用法：
    python generators/stylegan_generate.py \
        --checkpoint <network-snapshot-005000.pkl> \
        --outdir <workspace/generated/stylegan4sar/<variant>> \
        --angles-json <workspace/real/128/train/labels.json> \
        --per-class 100 --aasg-enabled 1 --device cuda

输出：
    cls{ci:02d}_az{az:06.1f}_s{seed}.png  （灰度 PNG，取 RGB 第一通道，幅度 uint8）
    generation_manifest.json              （每张：class_idx/class_name/azimuth_deg/seed/filename）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# stylegan4SAR 源码目录（dnnlib / legacy 所在）
_SG_DIR = Path(__file__).resolve().parents[1]  # 由 stage_b 以 cwd 传入更稳妥，这里兜底
if (Path(os.getcwd()) / "legacy.py").exists():
    sys.path.insert(0, os.getcwd())
else:
    for cand in ["../stylegan4SAR", "sar_models/stylegan4SAR"]:
        p = Path(cand)
        if (p / "legacy.py").exists():
            sys.path.insert(0, str(p.resolve()))
            break

import dnnlib  # noqa: E402
import legacy  # noqa: E402


def sample_spread_angles(angles_deg, n):
    """从真实角度列表里挑 n 个尽量分散的角度（用于 FR 精确配对）。"""
    angles = sorted(float(a) for a in angles_deg)
    if n >= len(angles):
        return angles[:n]
    # 均匀取 n 个下标
    idx = [int(round(i * (len(angles) - 1) / (n - 1))) for i in range(n)]
    return [angles[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--angles-json", required=True, help="Stage A labels.json（真实角度列表）")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--aasg-enabled", type=int, choices=[0, 1], default=1)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 读真实角度，按类分组
    labels = json.load(open(args.angles_json, encoding="utf-8"))
    per_class_angles = [[] for _ in range(args.classes)]
    for lb in labels:
        per_class_angles[int(lb["class_idx"])].append(float(lb["azimuth_deg"]))

    # 加载生成器
    print(f"[stylegan] loading {args.checkpoint}")
    with open(args.checkpoint, "rb") as f:
        net = legacy.load_network_pkl(f)["G_ema"].to(device).eval()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    n_total = 0
    for ci in range(args.classes):
        angles = sample_spread_angles(per_class_angles[ci], args.per_class)
        n = len(angles)
        for start in range(0, n, args.batch):
            chunk = angles[start:start + args.batch]
            B = len(chunk)
            z = torch.randn([B, net.z_dim], device=device)
            c = torch.zeros([B, args.classes + 2], dtype=torch.float32, device=device)
            for j, az in enumerate(chunk):
                c[j, ci] = 1.0
                c[j, args.classes] = math.sin(math.radians(az))
                c[j, args.classes + 1] = math.cos(math.radians(az))
            with torch.no_grad():
                gen = net(z, c, noise_mode="const", aasg_enabled=bool(args.aasg_enabled))
            # [B,3,H,W] -> 灰度 uint8（取第一通道）
            arr = gen[:, 0:1].clamp(-1, 1).add(1).mul(127.5).round().clamp(0, 255)
            arr = arr.squeeze(1).cpu().numpy().astype(np.uint8)  # (B,H,W)
            for j, az in enumerate(chunk):
                fname = f"cls{ci:02d}_az{az:06.1f}_s{args.seed + start + j}.png"
                Image.fromarray(arr[j], mode="L").save(outdir / fname)
                manifest.append({
                    "class_idx": ci,
                    "class_name": None,  # 由 stage_b 回填类名
                    "azimuth_deg": az,
                    "seed": args.seed + start + j,
                    "filename": fname,
                })
                n_total += 1
        print(f"[stylegan] class {ci}: generated {n} images")

    (outdir / "generation_manifest.json").write_text(
        json.dumps({"total": n_total, "per_class": args.per_class,
                    "aasg_enabled": bool(args.aasg_enabled), "samples": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stylegan] done: {n_total} images -> {outdir}")


if __name__ == "__main__":
    main()
