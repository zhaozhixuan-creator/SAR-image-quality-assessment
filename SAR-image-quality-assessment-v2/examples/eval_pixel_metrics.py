#!/usr/bin/env python3
"""在 mstar_real 上验证 7 个新增 FR 指标（MSE/RMSE/PSNR/NCC/UQI/MS-SSIM/FSIM）。

两列对照设计（与 eval_4_metrics.py 一致）：
- real-real2（自洽）：近乎相同的第二次观测，应接近理想值 → 验证实现正确。
- real-fake（退化）：施加模糊/噪声/再散斑退化，应明显偏离理想值 → 验证指标可区分质量。

运行：
    python examples/eval_pixel_metrics.py
"""
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
    import gan_metrics as gm

    real = load(DATA, "real_")
    real2 = load(DATA, "real2_")
    fake = load(DATA, "fake_")
    n = len(real)
    assert n == len(real2) == len(fake)
    print(f"读入 real/real2/fake 各 {n} 张")

    metrics = [
        ("MSE", gm.mse, "自洽≈0 / 退化>0"),
        ("RMSE", gm.rmse, "自洽≈0 / 退化>0"),
        ("PSNR", gm.psnr, "自洽高 / 退化更低"),
        ("NCC", gm.ncc, "自洽≈1 / 退化更低"),
        ("UQI", gm.uqi, "自洽≈1 / 退化更低"),
        ("MS-SSIM", gm.ms_ssim, "自洽≈1 / 退化更低"),
        ("FSIM", gm.fsim, "自洽≈1 / 退化更低"),
    ]

    print("\n================ 新增 7 项 FR 指标（mstar_real 验证） ================")
    print(f"{'指标':<9}{'real-real2(自洽)':>18}{'real-fake(退化)':>18}    期望")
    results = {}
    for name, fn, expect in metrics:
        v_self = fn(real, real2)
        v_degr = fn(real, fake)
        results[name] = {"self": float(v_self), "degrade": float(v_degr)}
        print(f"{name:<9}{v_self:>18.4f}{v_degr:>18.4f}    {expect}")
    print("=====================================================================")

    out = os.path.join(DATA, "pixel_metrics_validation.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
