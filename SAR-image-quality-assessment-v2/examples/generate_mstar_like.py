#!/usr/bin/env python3
"""生成仿真 MSTAR-like 目标切片（128×128 单通道幅度），用于验证生成模型评估指标。

没有真实 MSTAR 数据时的替身：把一个「车载坐标系」的非对称散射点布局按方位角旋转，
叠加 L 视乘性斑点背景，得到不同方位角的 SAR 目标切片。

- 训练集：全角度覆盖的真实切片（供方位角估计器 R 与 ASC 提取器 E_asc 预训练）。
- 评估集「真实」：干净切片。
- 评估集「生成」：施加退化（轻微模糊 + 加噪 + 视数偏移 + 方位角偏移）的切片，
  模拟生成器输出（含角度控制误差）。

输出：<out>/train_real.npy、train_angles.npy、real_<k>.npy、fake_<k>.npy、angles.json。
"""
import argparse
import json
import os

import numpy as np


# ---- 目标散射点布局（车载坐标系，x 指向炮管方向，单位像素 @128 尺度）----
def target_scatterers():
    return [
        # 炮管（向右延伸，确定朝向）
        (6, 0, 1.00), (10, 0, 0.95), (14, 0, 0.85), (18, 0, 0.75),
        # 车体（不对称：右侧更长）
        (-4, -3, 0.70), (-4, 3, 0.70), (0, -3, 0.78), (0, 3, 0.78),
        (4, -3, 0.82), (4, 3, 0.82),
        # 炮塔（偏右）
        (2, 0, 1.00), (3, -1, 0.60), (3, 1, 0.60),
        # 尾舱（左侧）
        (-7, 0, 0.70),
    ]


def render_chip(angle_deg, size=128, looks=4.0, clutter=1.0, seed=0,
                peak=6000.0, width=1.5):
    """按方位角渲染一张目标切片（幅度域，float32）。"""
    rng = np.random.RandomState(seed)
    # 背景：均匀杂波 + L 视乘性斑点（强度域）
    if looks == 1:
        speckle = rng.exponential(1.0, size=(size, size))
    else:
        speckle = rng.gamma(looks, 1.0 / looks, size=(size, size))
    intensity = clutter * speckle.astype(np.float64)

    # 目标散射点：旋转后叠加截断 sinc² 点响应（强度域）
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    yy, xx = np.mgrid[0:size, 0:size]
    for (lx, ly, a) in target_scatterers():
        xr = c * lx - s * ly
        yr = s * lx + c * ly
        px = size / 2.0 + xr
        py = size / 2.0 + yr
        dx = (xx - px) / width
        dy = (yy - py) / width
        resp = np.sinc(dx) * np.sinc(dy)
        support = (np.abs(dx) < 3.0) & (np.abs(dy) < 3.0)
        resp = np.where(support, resp, 0.0)
        intensity += (a * a * peak) * (resp ** 2)

    return np.sqrt(np.maximum(intensity, 0.0)).astype(np.float32)


def degrade(amp, blur_sigma=0.3, noise_frac=0.02, seed=0,
            target_h=0.4, target_w=0.4):
    """轻微退化：仅对目标区施加高斯模糊 + 加性高斯噪声（模拟生成器目标失真）。

    背景保持纯斑点，其统计差异由 looks_fake 单独体现，避免模糊污染背景、
    抵消 ENL/BVE 对视数差异的响应。
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    H, W = amp.shape
    th, tw = int(round(H * target_h)), int(round(W * target_w))
    y0, x0 = (H - th) // 2, (W - tw) // 2
    blurred = gaussian_filter(amp, sigma=blur_sigma)
    noisy = blurred + noise_frac * float(amp.std()) * rng.randn(*amp.shape).astype(np.float32)
    out = amp.copy()
    out[y0:y0 + th, x0:x0 + tw] = noisy[y0:y0 + th, x0:x0 + tw]
    return np.maximum(out, 0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="评估集每类图像数")
    ap.add_argument("--train-n", type=int, default=360, help="R/E_asc 训练图像数（全角度）")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="mstar_like")
    ap.add_argument("--angle-offset", type=float, default=10.0, help="生成集方位角偏移（度）")
    ap.add_argument("--looks-real", type=float, default=4.0)
    ap.add_argument("--looks-fake", type=float, default=3.2)
    ap.add_argument("--blur-sigma", type=float, default=0.4)
    ap.add_argument("--noise-frac", type=float, default=0.03)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 训练用真实切片（全角度 0..360，供 R 与 E_asc 预训练）
    train_angles = np.linspace(0.0, 360.0, args.train_n, endpoint=False)
    train_real = np.stack([render_chip(a, size=args.size, looks=args.looks_real,
                                       seed=1000 + k) for k, a in enumerate(train_angles)])
    np.save(os.path.join(args.out, "train_real.npy"), train_real)
    np.save(os.path.join(args.out, "train_angles.npy"), train_angles)

    # 评估集角度：含近 0°/360° 边界角度，验证循环距离
    rng = np.random.RandomState(0)
    boundary = np.array([1.0, 3.0, 357.0, 359.0])
    rest = np.sort(rng.choice(360, max(args.n - len(boundary), 0), replace=False)).astype(float)
    eval_angles = np.sort(np.concatenate([boundary, rest]))

    real, real2, fake = [], [], []
    for k, a in enumerate(eval_angles):
        real.append(render_chip(a, size=args.size, looks=args.looks_real, seed=2000 + k))
        # 自洽性参照：同角度、不同斑点实现的第二张干净图
        real2.append(render_chip(a, size=args.size, looks=args.looks_real, seed=2500 + k))
        # 「被要求角度 a」但实际渲染 a+offset（模拟角度控制误差），并施加退化
        fake.append(degrade(render_chip(a + args.angle_offset, size=args.size,
                                        looks=args.looks_fake, seed=3000 + k),
                           blur_sigma=args.blur_sigma, noise_frac=args.noise_frac, seed=4000 + k))

    for k in range(len(eval_angles)):
        np.save(os.path.join(args.out, f"real_{k:03d}.npy"), real[k])
        np.save(os.path.join(args.out, f"real2_{k:03d}.npy"), real2[k])
        np.save(os.path.join(args.out, f"fake_{k:03d}.npy"), fake[k])
    with open(os.path.join(args.out, "angles.json"), "w") as fh:
        json.dump({"angles": [float(a) for a in eval_angles],
                   "angle_offset": args.angle_offset}, fh)

    print(f"已生成 {args.out}/：train_real {len(train_real)} 张、"
          f"real/fake 各 {len(eval_angles)} 张（{args.size}×{args.size} 幅度）。")


if __name__ == "__main__":
    main()
