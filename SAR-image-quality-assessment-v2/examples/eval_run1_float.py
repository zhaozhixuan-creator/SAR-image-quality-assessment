#!/usr/bin/env python3
"""在 run1_float（自己的生成集）上评估 7 项 FR 指标，以完整 MSTAR 作参考。

配对方式：同类交叉全配对取平均——每张生成图与同类别所有真实 MSTAR 图
逐张算指标后取平均，再对所有生成图取平均。

预处理（统一到 [0,1] 灰度域，与 build_mstar_ref.py 的参考集配套）：
- 生成图（run1_float 模型输出 [-1,1]）：(x+1)/2  → [0,1]（等价 png 灰度映射）
- 真实图（mstar_ref 原始幅度）：x / gmax_real  → [0,1]（gmax_real = 参考集全局 max）

运行：
    python examples/eval_run1_float.py                      # 全量（80 生成 × 72 参考）
    python examples/eval_run1_float.py --max-ref-per-class 8   # 加速测试
    python examples/eval_run1_float.py --skip-slow          # 跳过 MS-SSIM/FSIM
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN1 = r"C:\Users\Lenovo\Desktop\run1_float"
REF = "mstar_ref"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", default=RUN1)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--max-ref-per-class", type=int, default=0,
                    help="每类最多用多少张参考图（0=全部 72 张，用于加速）")
    ap.add_argument("--skip-slow", action="store_true", help="跳过 MS-SSIM/FSIM（较慢）")
    args = ap.parse_args()

    import gan_metrics as gm

    # 加载生成集
    gen = np.load(os.path.join(args.run1, "images_float_neg1_1.npy"))
    gen = gen[..., 0] if gen.ndim == 4 else gen
    gen_labels = np.load(os.path.join(args.run1, "labels.npy"))
    print(f"生成集：{gen.shape}，标签 {len(gen_labels)}，值域 [{gen.min():.3f}, {gen.max():.3f}]")

    # 加载参考集
    ref = np.load(os.path.join(args.ref, "ref_images.npy"))
    ref_labels = np.load(os.path.join(args.ref, "ref_labels.npy"))
    with open(os.path.join(args.ref, "class_names.json")) as fh:
        class_names = json.load(fh)
    print(f"参考集：{ref.shape}，标签 {len(ref_labels)}，类名 {class_names}")

    # 预处理到 [0,1]
    gen01 = (gen.astype(np.float64) + 1.0) / 2.0
    gmax_real = float(ref.max())
    ref01 = ref.astype(np.float64) / max(gmax_real, 1e-9)
    print(f"预处理：生成图 (x+1)/2 → [0,1]；真实图 x/{gmax_real:.2f} → [0,1]")

    metrics = [
        ("MSE", gm.mse), ("RMSE", gm.rmse), ("PSNR", gm.psnr),
        ("NCC", gm.ncc), ("UQI", gm.uqi), ("MS-SSIM", gm.ms_ssim), ("FSIM", gm.fsim),
    ]
    if args.skip_slow:
        metrics = metrics[:5]

    # 按类组织参考图索引
    ref_by_class = {c: [i for i, l in enumerate(ref_labels) if l == c]
                    for c in range(len(class_names))}
    if args.max_ref_per_class > 0:
        rng = np.random.RandomState(0)
        for c in ref_by_class:
            if len(ref_by_class[c]) > args.max_ref_per_class:
                ref_by_class[c] = sorted(rng.choice(
                    ref_by_class[c], args.max_ref_per_class, replace=False).tolist())

    # 逐生成图 → 同类所有参考 → 平均
    n_gen = len(gen01)
    per_gen = {name: np.zeros(n_gen) for name, _ in metrics}
    for gi in range(n_gen):
        L = int(gen_labels[gi])
        idxs = ref_by_class[L]
        g_list = [gen01[gi]] * len(idxs)
        r_list = [ref01[i] for i in idxs]
        for name, fn in metrics:
            per_gen[name][gi] = fn(g_list, r_list)
        if (gi + 1) % 10 == 0 or gi + 1 == n_gen:
            print(f"  已处理 {gi+1}/{n_gen} 张生成图", flush=True)

    # 汇总：总体均值 + 每类均值
    results = {"overall": {}, "per_class": {}}
    print("\n================ run1_float 生成集 FR 指标（同类交叉配对） ================")
    print(f"{'指标':<9}{'总体均值':>12}" + "".join(f"{c:>10}" for c in class_names))
    for name, _ in metrics:
        overall = float(per_gen[name].mean())
        results["overall"][name] = overall
        row = f"{name:<9}{overall:>12.4f}"
        results["per_class"][name] = {}
        for c in range(len(class_names)):
            mask = gen_labels == c
            v = float(per_gen[name][mask].mean()) if mask.sum() else float("nan")
            results["per_class"][name][class_names[c]] = v
            row += f"{v:>10.4f}"
        print(row)
    print("==========================================================================")

    out = os.path.join(args.run1, "pixel_metrics_results.json")
    with open(out, "w") as fh:
        json.dump({"class_names": class_names, "metrics": results}, fh, indent=2)
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
