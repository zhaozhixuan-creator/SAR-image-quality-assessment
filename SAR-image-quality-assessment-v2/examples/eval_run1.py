#!/usr/bin/env python3
"""对用户自生成的 SAR 数据集 run1_float 做论文 §3.1.4 前四项指标的质检评估。

数据说明
--------
run1_float/ 是 DDIM 扩散模型（checkpoint=checkpoints/mstar_pdm，1000 步，DDIM 采样）
在 MSTAR 上训练后生成的 SAR 图像：10 类 × 每类 8 张 = 80 张，128×128 单通道。
   - images_float_neg1_1.npy：模型输出（clip_denoised 后、uint8 量化前），值域 [-1, 1]
   - labels.npy：类别 0..9（顺序 2S1/BMP2/BRDM2/BTR60/BTR70/D7/T62/T72/ZIL131/ZSU234）
   - manifest.json 里 png_mapping = clip((x+1)*127.5, 0, 255)

域对齐（关键）
------------
论文四指标里 FID/SSIM/AFS 各自归一化（联合 gmax 或逐图 gmax）、ΔENL 用强度域
ENL=μ²/σ² 是尺度不变比，故绝对尺度不影响；但 ENL 对「平移」敏感，负值会破坏 μ²/σ²。
因此把模型输出的 [-1, 1] 按模型自身的线性映射还原到非负幅度域：
        magnitude m = clip((x + 1) / 2, 0, 1)
真实 MSTAR 幅度沿用 mstar_real/real_*.npy 的归一化幅度（≈[0, 1.16]）。
ΔENL 在强度域（幅度平方）计算，与论文式(41) / eval_4_metrics.py 口径一致。

配对（SSIM/AFS/ΔENL 需逐对）
----------------------------
生成图 labels.npy 只含类别、无方位角 → 无法按角度配对（扩散模型为类别条件生成）。
故采用「类匹配」：每张生成图与同类别的一张真实图按序配对（每类 8 对 × 10 类 = 80 对）。
真实评估集 real_<k>.npy 的类顺序与生成图一致（real_[c*10+j] 为第 c 类第 j 张），
由 mstar_to_chips.py 的确定性选择保证。

运行：
    python -u examples/eval_run1.py [--gen-dir ...] [--asc-epochs 25] [--no-fid]
输出：控制台 + 当前目录下 run1_eval_report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gan_metrics as gm

# 类别显示名（与 mstar_to_chips.py / run1_float manifest 顺序一致）
CLASS_NAMES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
               "D7", "T62", "T72", "ZIL131", "ZSU234"]

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mstar_real")


def load_generated(gen_dir: str):
    """载入生成图，返回 (mag[list], labels[array])，mag 为非负幅度域 [0,1]。"""
    im = np.load(os.path.join(gen_dir, "images_float_neg1_1.npy"))
    if im.ndim == 4:
        im = im[:, :, :, 0]
    labels = np.load(os.path.join(gen_dir, "labels.npy"))
    mag = [np.clip((x + 1.0) / 2.0, 0.0, 1.0).astype(np.float32) for x in im]
    return mag, labels


def load_real(real_dir: str, n_per_class: int = 10):
    """载入真实评估集，返回 real_by_class[class_idx] -> list[mag]（每类 10 张）。"""
    n_cls = len(CLASS_NAMES)
    real_by_class = [[] for _ in range(n_cls)]
    for k in range(n_cls * n_per_class):
        r = np.load(os.path.join(real_dir, f"real_{k:03d}.npy"))
        real_by_class[k // n_per_class].append(r)
    return real_by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=r"C:\Users\Lenovo\Desktop\run1_float")
    ap.add_argument("--real-dir", default=REAL_DIR)
    ap.add_argument("--asc-epochs", type=int, default=25)
    ap.add_argument("--fid-batch", type=int, default=8)
    ap.add_argument("--no-fid", action="store_true", help="跳过 FID（CPU 上较慢）")
    args = ap.parse_args()

    # ---- 载入 ----
    gen_mag, labels = load_generated(args.gen_dir)
    real_by_class = load_real(args.real_dir)
    n_cls = len(CLASS_NAMES)

    # 类别分组
    gen_by_class = [[] for _ in range(n_cls)]
    for i, c in enumerate(labels):
        gen_by_class[int(c)].append(gen_mag[i])

    print(f"生成图 {len(gen_mag)} 张（{n_cls} 类 × 每类 8），真实图每类 10 张", flush=True)
    per_cls = [len(gen_by_class[c]) for c in range(n_cls)]
    assert all(n == 8 for n in per_cls), f"每类生成数应为 8，实际 {per_cls}"

    # 类匹配配对：gen_by_class[c][j] <-> real_by_class[c][j]，j=0..7
    gen_match, real_match = [], []
    for c in range(n_cls):
        for j in range(len(gen_by_class[c])):
            gen_match.append(gen_by_class[c][j])
            real_match.append(real_by_class[c][j])
    N = len(gen_match)
    print(f"类匹配配对 {N} 对", flush=True)

    # ---- 1) 预训练 ASC 提取器 E_asc（AFS 用）----
    print("\n[1/2] 预训练 ASC 提取器 E_asc（AFS 用）…", flush=True)
    train_real = [x for x in np.load(os.path.join(args.real_dir, "train_real.npy"))]
    E = gm.train_asc_extractor(train_real, epochs=args.asc_epochs)

    # ---- 2) 计算四项指标 ----
    print("\n[2/2] 计算前四项指标（FID / SSIM / AFS / ΔENL）…", flush=True)

    # 强度域（ΔENL）
    gen_i = [m ** 2 for m in gen_match]
    real_i = [r ** 2 for r in real_match]

    # 全体真实（FID 分布距离用，100 张）
    all_real = [r for c in range(n_cls) for r in real_by_class[c]]

    report = {}
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("\n================ 前四项指标（生成图 vs 真实图） ================")
    emit(f"{'指标':<6}{'全局值':>12}   说明")

    # FID
    if not args.no_fid:
        fid_g = gm.fid(all_real, gen_mag, batch=args.fid_batch)
        report["FID"] = fid_g
        emit(f"{'FID':<6}{fid_g:>12.2f}   ↓越低越好，分布距离（生成 80 vs 真实 100）")
    else:
        emit(f"{'FID':<6}{'（跳过）':>12}")

    # SSIM / AFS / ΔENL
    ssim_g = gm.ssim(real_match, gen_match)
    afs_g = gm.afs(real_match, gen_match, E)
    denl_g = gm.delta_enl(real_i, gen_i)
    report["SSIM"] = ssim_g
    report["AFS"] = afs_g
    report["ΔENL"] = denl_g
    emit(f"{'SSIM':<6}{ssim_g:>12.4f}   ↑越高越好（类匹配，含方位角错位影响）")
    emit(f"{'AFS':<6}{afs_g:>12.4f}   ↑越高越好（目标区 ASC 特征余弦相似度）")
    emit(f"{'ΔENL':<6}{denl_g:>12.4f}   ↓越低越好（背景区强度 ENL 差）")
    emit("=============================================================")

    # ---- 3) 分指标、分指标与参考基准 ----
    emit("\n---- 分指标统计（各类别均值） ----")
    emit(f"{'类别':<8}{'FID':>9}{'SSIM':>8}{'AFS':>8}{'ΔENL':>9}")
    per_cls_stats = []
    for c in range(n_cls):
        g_c = gen_by_class[c]
        r_c = real_by_class[c]
        g_i = [m ** 2 for m in g_c]
        r_i = [r ** 2 for r in r_c]
        rc = real_by_class[c][:len(g_c)]  # 取前 8 张配对
        if not args.no_fid:
            fid_c = gm.fid(r_c, g_c, batch=args.fid_batch)
        else:
            fid_c = float("nan")
        ssim_c = gm.ssim(rc, g_c)
        afs_c = gm.afs(rc, g_c, E)
        denl_c = gm.delta_enl(r_i, g_i)
        per_cls_stats.append((CLASS_NAMES[c], fid_c, ssim_c, afs_c, denl_c))
        emit(f"{CLASS_NAMES[c]:<8}{fid_c:>9.2f}{ssim_c:>8.4f}{afs_c:>8.4f}{denl_c:>9.4f}")

    # ---- 4) 写报告 ----
    out_path = os.path.join(os.getcwd(), "run1_eval_report.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# run1_float 生成 SAR 图像质检评估报告\n\n")
        fh.write("## 0. 数据与口径\n\n")
        fh.write("- 生成模型：DDIM 扩散模型（`checkpoints/mstar_pdm`，1000 步）\n")
        fh.write("- 生成图：10 类 × 每类 8 = 80 张，128×128 单通道，模型输出 [-1,1] "
                 "→ 非负幅度域 `m=clip((x+1)/2,0,1)`\n")
        fh.write("- 真实参照：MSTAR（GitHub jwcalder/MSTAR-Active-Learning），"
                 "15° 俯角、72 方位角/类，评估集每类 10 张，归一化幅度 ≈[0,1.16]\n")
        fh.write("- ΔENL 在强度域（幅度平方）计算，ENL=μ²/σ²，与论文式(41) 一致\n")
        fh.write("- 配对：生成图无方位角标签，采用**类匹配**（同类别按序配对）\n\n")
        fh.write("## 1. 全局结果\n\n")
        fh.write("| 指标 | 值 | 方向 |\n|---|---|---|\n")
        if not args.no_fid:
            fh.write(f"| FID | {report['FID']:.2f} | ↓越低越好 |\n")
        fh.write(f"| SSIM | {report['SSIM']:.4f} | ↑越高越好 |\n")
        fh.write(f"| AFS | {report['AFS']:.4f} | ↑越高越好 |\n")
        fh.write(f"| ΔENL | {report['ΔENL']:.4f} | ↓越低越好 |\n\n")
        fh.write("## 2. 分指标统计\n\n")
        fh.write("| 类别 | FID | SSIM | AFS | ΔENL |\n|---|---|---|---|---|\n")
        for name, fid_c, ssim_c, afs_c, denl_c in per_cls_stats:
            fh.write(f"| {name} | {fid_c:.2f} | {ssim_c:.4f} | {afs_c:.4f} | {denl_c:.4f} |\n")
        fh.write("\n## 3. 解读\n\n")
        fh.write("（见下方对话中的完整解读）\n")
    emit(f"\n报告已写入 {out_path}", )


if __name__ == "__main__":
    main()
