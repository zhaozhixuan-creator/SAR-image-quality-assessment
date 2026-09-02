#!/usr/bin/env python3
"""仅计算论文 §3.1.4 的前四项指标：FID / SSIM / AFS / ΔENL。

与 eval_gan_metrics.py（六项全算，需预训练方位角估计器 R，仅 CMAE 用到）不同，
本脚本聚焦前四项，跳过 R 训练，只预训练 ASC 提取器 E_asc（AFS 需要），
在纯 CPU 环境几分钟即可出结果。

运行：
    python examples/eval_4_metrics.py
    python examples/eval_4_metrics.py --no-fid   # 跳过较慢的 Inception FID

输出两列对照：
- real-real2（自洽）：同角度、近乎相同的第二次观测，应接近理想值 → 验证实现正确。
- real-fake（退化）：同类别、角度 +offset 的切片再退化，应明显偏离理想值 → 验证指标可区分质量。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = "mstar_real"


def load(dirpath, prefix):
    names = sorted(f for f in os.listdir(dirpath)
                   if f.startswith(prefix) and f.endswith(".npy"))
    return [np.load(os.path.join(dirpath, f)) for f in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asc-epochs", type=int, default=25)
    ap.add_argument("--fid-batch", type=int, default=8)
    ap.add_argument("--no-fid", action="store_true", help="跳过 FID（CPU 上 Inception 较慢）")
    args = ap.parse_args()

    import gan_metrics as gm

    real = load(DATA, "real_")
    real2 = load(DATA, "real2_")
    fake = load(DATA, "fake_")
    with open(os.path.join(DATA, "angles.json")) as fh:
        meta = json.load(fh)
    angles = np.asarray(meta["angles"])
    offset = meta.get("angle_offset", 10.0)
    n = len(real)
    assert n == len(real2) == len(fake), "real/real2/fake 数量不一致"

    train_real = [x for x in np.load(os.path.join(DATA, "train_real.npy"))]
    print(f"读入评估集 real/real2/fake 各 {n} 张；训练集（真实，仅用于预训练 E_asc）{len(train_real)} 张",
          flush=True)

    # 1) 预训练 ASC 提取器 E_asc（AFS 用），重构自监督，秒级~分钟级
    print("\n[1/2] 预训练 ASC 提取器 E_asc（AFS 用）…", flush=True)
    E = gm.train_asc_extractor(train_real, epochs=args.asc_epochs)

    # 2) 计算前四项指标
    print("\n[2/2] 计算前四项指标（FID / SSIM / AFS / ΔENL）…", flush=True)

    # ΔENL 在强度域（幅度平方）计算，ENL=μ²/σ² 的经典定义域
    real_i = [r ** 2 for r in real]
    real2_i = [r ** 2 for r in real2]
    fake_i = [f ** 2 for f in fake]

    print("\n================ 前四项指标 ================", flush=True)
    print(f"{'指标':<7}{'real-real2(自洽)':>18}{'real-fake(退化)':>18}    期望", flush=True)
    if not args.no_fid:
        fid_self = gm.fid(real, real2, batch=args.fid_batch)
        fid_degr = gm.fid(real, fake, batch=args.fid_batch)
        print(f"{'FID':<7}{fid_self:>18.2f}{fid_degr:>18.2f}    自洽≈0 / 退化>0", flush=True)
    else:
        print(f"{'FID':<7}{'（跳过）':>18}{'（跳过）':>18}", flush=True)

    ssim_self = gm.ssim(real, real2)
    ssim_degr = gm.ssim(real, fake)
    print(f"{'SSIM':<7}{ssim_self:>18.4f}{ssim_degr:>18.4f}    自洽≈1 / 退化更低", flush=True)

    afs_self = gm.afs(real, real2, E)
    afs_degr = gm.afs(real, fake, E)
    print(f"{'AFS':<7}{afs_self:>18.4f}{afs_degr:>18.4f}    自洽≈1 / 退化<1", flush=True)

    denl_self = gm.delta_enl(real_i, real2_i)
    denl_degr = gm.delta_enl(real_i, fake_i)
    print(f"{'ΔENL':<7}{denl_self:>18.4f}{denl_degr:>18.4f}    自洽≈0 / 退化>0", flush=True)
    print("===========================================", flush=True)

    print("\n解读：", flush=True)
    print(" - real-real2（自洽）列：同角度、近乎相同的第二次观测，应接近理想值", flush=True)
    print("   （FID≈0 / SSIM≈1 / AFS≈1 / ΔENL≈0）→ 验证四指标实现正确。", flush=True)
    print(" - real-fake（退化）列：同类别、真实角度 = 要求角 + offset 的切片再施加", flush=True)
    print("   模糊/噪声/背景再散斑退化，应明显偏离理想值 → 验证指标能区分生成质量。", flush=True)
    print(f"   （退化施加的角度偏移 offset = {offset:g}°）", flush=True)


if __name__ == "__main__":
    main()
