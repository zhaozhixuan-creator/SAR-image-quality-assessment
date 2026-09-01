#!/usr/bin/env python3
"""端到端验证论文 §3.1.4 六项生成模型评估指标。

运行：
    python examples/generate_mstar_like.py        # 生成仿真 MSTAR-like 切片
    python examples/eval_gan_metrics.py           # 训练 R/E_asc 并计算六项指标

说明：R(·) 与 E_asc 是论文中的「预训练辅助网络」。本脚本在真实切片上按论文口径预训练
（R 仅在真实图上训练、评估时冻结；E_asc 以重构自监督训练）。真实 MSTAR 不可得，切片为
仿真替身，因此这里输出的是「实现正确性 + 指标行为」验证，数值不与论文 Table 4 直接可比。

输出两列对照：
- real-real2：自洽性检查（同角度、不同斑点的两张干净图），应接近理想值，验证实现正确。
- real-fake：退化图偏离程度，各项应明显偏离理想值。
CMAE 用 R 反推角度后与「被要求角度」比循环误差：退化图实际渲染角 = 要求角 + offset，
故 real-fake 的 CMAE ≈ offset，说明角度控制误差被正确捕捉。
"""
import argparse
import json
import os
import sys

import numpy as np

# 允许从 examples/ 下运行：把项目根目录加入导入路径，以 import gan_metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(dirpath, prefix):
    names = sorted(f for f in os.listdir(dirpath)
                   if f.startswith(prefix) and f.endswith(".npy"))
    return [np.load(os.path.join(dirpath, f)) for f in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mstar_like")
    ap.add_argument("--angle-epochs", type=int, default=40)
    ap.add_argument("--asc-epochs", type=int, default=25)
    ap.add_argument("--fid-batch", type=int, default=8)
    ap.add_argument("--no-fid", action="store_true", help="跳过 FID（CPU 上 Inception 较慢）")
    args = ap.parse_args()

    import gan_metrics as gm

    real = load(args.data, "real_")
    real2 = load(args.data, "real2_")
    fake = load(args.data, "fake_")
    with open(os.path.join(args.data, "angles.json")) as fh:
        meta = json.load(fh)
    angles = np.asarray(meta["angles"])
    offset = meta.get("angle_offset", 5.0)
    n = len(real)
    assert n == len(fake) == len(angles), "real/fake/angles 数量不一致"

    train_real = [x for x in np.load(os.path.join(args.data, "train_real.npy"))]
    train_angles = np.load(os.path.join(args.data, "train_angles.npy")).tolist()
    print(f"读入评估集 real/fake 各 {n} 张；训练集（真实）{len(train_real)} 张")

    # 1) 预训练方位角估计器 R（仅在真实图上）
    print("\n[1/3] 训练方位角估计器 R(·) …")
    R = gm.train_angle_estimator(train_real, train_angles, epochs=args.angle_epochs)

    # 2) 预训练 ASC 提取器 E_asc（真实目标切片，重构自监督）
    print("\n[2/3] 训练 ASC 提取器 E_asc …")
    E = gm.train_asc_extractor(train_real, epochs=args.asc_epochs)

    # 3) 计算六项指标
    print("\n[3/3] 计算指标 …")
    phi_real = gm.estimate_angles(real, R)
    phi_fake = gm.estimate_angles(fake, R)

    # ENL/BVE 按经典定义在强度域（功率）计算：幅度平方 → 强度，ENL ≈ 视数 L
    real_i = [r ** 2 for r in real]
    real2_i = [r ** 2 for r in real2]
    fake_i = [f ** 2 for f in fake]

    print("\n================ 六项指标 ================")
    print(f"{'指标':<7}{'real-real2(自洽)':>18}{'real-fake(退化)':>18}    期望")
    if not args.no_fid:
        print(f"{'FID':<7}{gm.fid(real, real2, batch=args.fid_batch):>18.2f}"
              f"{gm.fid(real, fake, batch=args.fid_batch):>18.2f}    自洽≈0 / 退化>0")
    else:
        print(f"{'FID':<7}{'（跳过）':>18}{'（跳过）':>18}")
    print(f"{'SSIM':<7}{gm.ssim(real, real2):>18.4f}{gm.ssim(real, fake):>18.4f}    自洽高 / 退化更低")
    print(f"{'AFS':<7}{gm.afs(real, real2, E):>18.4f}{gm.afs(real, fake, E):>18.4f}    自洽≈1 / 退化<1")
    print(f"{'ΔENL':<7}{gm.delta_enl(real_i, real2_i):>18.4f}{gm.delta_enl(real_i, fake_i):>18.4f}    自洽≈0 / 退化>0")
    print(f"{'BVE':<7}{gm.bve(real_i, real2_i):>18.4f}{gm.bve(real_i, fake_i):>18.4f}    自洽≈0 / 退化>0")

    cmae_self = gm.cmae(phi_real, angles)
    cmae_degr = gm.cmae(phi_fake, angles)
    print(f"{'CMAE°':<7}{cmae_self:>18.2f}{cmae_degr:>18.2f}    自洽≈0(估角) / 退化≈offset({offset:g})")
    print("==========================================")
    print("\n解读：")
    print(" - real-real2（自洽）列：同角度、不同斑点的两张干净图，接近理想值 → 实现正确。")
    print(" - real-fake（退化）列：明显偏离理想值 → 指标能区分生成质量。")
    print(" - CMAE 由 R 反推角度后与「被要求角度」比循环误差；")
    print("   退化图实际渲染角 = 要求角 + offset，故 CMAE≈offset 说明角度控制误差被正确捕捉。")


if __name__ == "__main__":
    main()
