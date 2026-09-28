#!/usr/bin/env python3
"""exp4 角度控制验证：条件方位角 vs R 预测方位角（基线 vs 定向微调）。

对同一固定 z、同一类别，扫条件方位角 [0..360)，用质检估计器 R 反推生成图朝向。
若生成器真正学会方位角控制，「条件角→预测角」应接近对角线 y=x；
基线（CMAE≈91°）则应接近水平线/噪声（预测角与条件角无关）。

在 stylegan4SAR venv 内运行（需 torch1.7 + matplotlib + legacy/dnnlib + 本仓 plugin）。

用法：
  python scripts/verify_angle_control.py \
    --baseline <network-snapshot-001300.pkl> --finetuned <network-snapshot-000200.pkl>
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

SG_DIR = "/home/zhaozhixuan/SAR-Generation/sar_models/stylegan4SAR"
if SG_DIR not in sys.path:
    sys.path.insert(0, SG_DIR)

from plugin.azimuth_loss import load_azimuth_estimator  # noqa: E402
import dnnlib  # noqa: E402
import legacy  # noqa: E402


def _to_r_input(gen: torch.Tensor) -> torch.Tensor:
    """[-1,1] (B,3,H,W) → [0,1] (B,1,H,W) 逐图 max 归一化（与 R 训练口径一致）。"""
    x = (gen[:, 0:1] + 1.0) * 0.5
    x = x.clamp(min=0.0)
    return x / x.amax(dim=[2, 3], keepdim=True).clamp(min=1e-6)


def predict_angle(R, gen: torch.Tensor) -> float:
    with torch.no_grad():
        sc = R(_to_r_input(gen))[0]  # (sin, cos)
    return float(np.mod(np.degrees(np.arctan2(sc[0].item(), sc[1].item())), 360.0))


def sweep(G, R, angles_deg, class_idx: int, n_classes: int, device) -> list[float]:
    torch.manual_seed(1234)  # 固定 z，保证「同一潜码、不同条件角」对照可复现
    z = torch.randn([1, G.z_dim], device=device)
    preds = []
    for az in angles_deg:
        c = torch.zeros([1, n_classes + 2], dtype=torch.float32, device=device)
        c[0, class_idx] = 1.0
        c[0, n_classes] = math.sin(math.radians(az))
        c[0, n_classes + 1] = math.cos(math.radians(az))
        with torch.no_grad():
            gen = G(z, c, noise_mode="const")
        preds.append(predict_angle(R, gen))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--R", default="/home/zhaozhixuan/SAR-image-quality-assessment/"
                                  "SAR-image-quality-assessment-v3/workspace/checkpoints/R_128.pt")
    ap.add_argument("--outdir", default="results/experiments")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--n-classes", type=int, default=5)
    ap.add_argument("--class-idx", type=int, default=0)
    ap.add_argument("--step", type=float, default=15.0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    R = load_azimuth_estimator(args.R, device)
    angles = np.arange(0.0, 360.0, args.step).tolist()

    results = {"step": args.step, "angles": angles, "class_idx": args.class_idx}
    for name, ckpt in [("baseline", args.baseline), ("finetuned", args.finetuned)]:
        print(f"[verify] {name}: {ckpt}")
        with open(ckpt, "rb") as f:
            G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()
        preds = sweep(G, R, angles, args.class_idx, args.n_classes, device)
        results[name] = {"predicted": preds}
        # 循环误差统计
        err = [min(abs(a - p), 360 - abs(a - p)) for a, p in zip(angles, preds)]
        results[name]["cmae_deg"] = float(np.mean(err))
        results[name]["corr"] = float(np.corrcoef(angles, preds)[0, 1])
        print(f"  cmae={results[name]['cmae_deg']:.1f}°  corr={results[name]['corr']:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exp4_angle_control.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 散点图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    _fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(_fp):
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 360], [0, 360], "k--", lw=1, alpha=0.6, label="y=x (理想方位角控制)")
    ax.scatter(angles, results["baseline"]["predicted"], s=40, c="#d1495b",
               label=f"基线 (CMAE={results['baseline']['cmae_deg']:.0f}°)")
    ax.scatter(angles, results["finetuned"]["predicted"], s=40, c="#2e7d32",
               label=f"方位角定向微调 (CMAE={results['finetuned']['cmae_deg']:.0f}°)")
    ax.set_xlabel("条件方位角 (deg)")
    ax.set_ylabel("R 预测方位角 (deg)")
    ax.set_title("质检估计器 R 验证：条件方位角 vs 生成图朝向")
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "exp4_angle_control.png", dpi=130)
    print(f"[verify] saved {outdir / 'exp4_angle_control.png'}")


if __name__ == "__main__":
    main()
