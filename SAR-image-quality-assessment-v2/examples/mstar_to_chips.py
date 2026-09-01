#!/usr/bin/env python3
"""把真实 MSTAR 数据转成 v2 切片口径，替换仿真替身。

数据来源：GitHub 仓库 jwcalder/MSTAR-Active-Learning 的 Data/SAR10{a,b,c}.npz，
是 SDMS 官方公开 MSTAR mixed-targets 的预处理结果（88×88 幅度+相位 float 切片）。

默认转换全部 10 类（2S1/BMP2/BRDM2/BTR60/BTR70/D7/T62/T72/ZIL131/ZSU23-4），
`--classes paper` 则只取论文 §3.1.1 的 5 类（2S1/BRDM2/D7/T62/ZIL131）。
  - 每类 72 个方位角（0–355°，5° 间隔，最近邻匹配）
  - 固定俯角（默认 17°，SOC）
  - 幅度切片 resize 到 128×128

输出与 generate_mstar_like.py 相同的目录结构（默认 mstar_real/）：
  train_real.npy / train_angles.npy   —— 72×类数 张（10 类=720 / 5 类=360），供 R(·) 与 E_asc 预训练
  real_<k>.npy                        —— 干净真实切片（评估集）
  real2_<k>.npy                       —— 同一切片加 2% 乘性扰动（“第二次观测”自洽参照）
  fake_<k>.npy                        —— 同类别、角度 +offset 的切片施加退化
  angles.json                         —— 评估集“被要求角度” + angle_offset

说明：真实 MSTAR 每角度只有一次观测，无法像仿真那样换斑点种子生成 real2，
故 real2 用轻微乘性扰动模拟“近乎相同的第二次观测”，仅用于验证指标自洽性。
fake 用「同类别、真实角度 = 要求角度 + offset 的切片」+ 退化，模拟生成器角度控制误差。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 全部 10 类：MSTAR 实际 TargetType 字符串（SDMS 命名）→ 显示名
MSTAR_CLASSES = {
    "2s1_gun": "2S1",
    "bmp2_tank": "BMP2",
    "brdm2_truck": "BRDM2",
    "btr60_transport": "BTR60",
    "btr70_transport": "BTR70",
    "d7_bulldozer": "D7",
    "t62_tank": "T62",
    "t72_tank": "T72",
    "zil131_truck": "ZIL131",
    "zsu23-4_gun": "ZSU23-4",  # 用户所列 ZSU234 即 ZSU-23-4
}

# 论文 §3.1.1 仅用其中 5 类
PAPER_CLASSES = {k: MSTAR_CLASSES[k] for k in
                 ("2s1_gun", "brdm2_truck", "d7_bulldozer", "t62_tank", "zil131_truck")}


def norm_class(ttype: str) -> str:
    """目标类别字符串归一化：去下划线/空白/非字母数字，转小写。"""
    import re
    return re.sub(r"[^a-z0-9]", "", str(ttype).strip().lower())


def load_all(raw_dir):
    hdr, mag, phase = [], [], []
    for name in ["SAR10a.npz", "SAR10b.npz", "SAR10c.npz"]:
        p = os.path.join(raw_dir, name)
        if not os.path.exists(p):
            print(f"[warn] 未找到 {p}，跳过")
            continue
        d = np.load(p, allow_pickle=True)
        hdr.append(d["hdr"])
        mag.append(d["mag"])
        phase.append(d["phase"])
    if not hdr:
        raise SystemExit("[error] 没有读到任何 .npz，请先下载到 mstar_raw/")
    return (np.concatenate(hdr, axis=0),
            np.concatenate(mag, axis=0),
            np.concatenate(phase, axis=0))


def select_grid(angles, grid_step=5.0):
    """为每个 5° 网格点选一个最近的样本（去重），返回 (选中角度, 选中原始索引)。"""
    grid = np.arange(0, 360, grid_step)
    used = np.zeros(len(angles), dtype=bool)
    sel_ang, sel_idx = [], []
    for g in grid:
        d = np.abs((angles - g + 180.0) % 360.0 - 180.0)  # 循环距离
        d[used] = np.inf
        j = int(np.argmin(d))
        if np.isinf(d[j]):
            continue
        used[j] = True
        sel_ang.append(float(angles[j]))
        sel_idx.append(j)
    return np.asarray(sel_ang), np.asarray(sel_idx, dtype=int)


def resize_128(im):
    from scipy.ndimage import zoom
    if im.shape == (128, 128):
        return im.astype(np.float32)
    z = (128.0 / im.shape[0], 128.0 / im.shape[1])
    return zoom(im, z, order=1).astype(np.float32)


def second_look(amp, seed, frac=0.02):
    """轻微乘性扰动，模拟近乎相同的第二次观测。"""
    rng = np.random.RandomState(seed)
    return (amp * (1.0 + frac * rng.randn(*amp.shape).astype(np.float32))).astype(np.float32)


def degrade_real(amp, blur_sigma=0.3, noise_frac=0.03, looks_fake=3.0, seed=0):
    """退化：目标区模糊+加噪，背景区再散斑（强度域）以降低 ENL。"""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    H, W = amp.shape
    th, tw = int(round(H * 0.4)), int(round(W * 0.4))
    y0, x0 = (H - th) // 2, (W - tw) // 2

    # 目标区：高斯模糊 + 加性高斯噪声（幅度域）
    blurred = gaussian_filter(amp, sigma=blur_sigma)
    out = amp.copy()
    out[y0:y0 + th, x0:x0 + tw] = (
        blurred[y0:y0 + th, x0:x0 + tw]
        + noise_frac * float(amp.std()) * rng.randn(th, tw).astype(np.float32)
    )

    # 背景区：单位均值 Gamma 乘性散斑（强度域），改变有效视数 ENL
    I = out.astype(np.float64) ** 2
    bg = rng.gamma(looks_fake, 1.0 / looks_fake, size=(H, W))
    mask_bg = np.ones((H, W), dtype=bool)
    mask_bg[y0:y0 + th, x0:x0 + tw] = False
    I[mask_bg] = I[mask_bg] * bg[mask_bg]
    return np.sqrt(np.maximum(I, 0.0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_raw"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_real"))
    ap.add_argument("--depression", type=int, default=17, help="固定俯角（17=SOC 训练集）")
    ap.add_argument("--classes", choices=["all", "paper"], default="all",
                    help="all=全部 10 类；paper=论文 5 类子集")
    ap.add_argument("--grid", type=float, default=5.0, help="方位角网格步长（度）")
    ap.add_argument("--eval-per-class", type=int, default=10)
    ap.add_argument("--angle-offset", type=float, default=10.0)
    ap.add_argument("--blur-sigma", type=float, default=0.3)
    ap.add_argument("--noise-frac", type=float, default=0.03)
    ap.add_argument("--looks-fake", type=float, default=3.0)
    args = ap.parse_args()

    hdr, mag, phase = load_all(args.raw)
    n = len(hdr)
    print(f"读入 {n} 张 MSTAR 切片（原始 {mag.shape[1]}×{mag.shape[2]}）")

    ttype = np.array([str(h[0]).strip().lower() for h in hdr])
    az = np.array([float(h[2]) for h in hdr])
    dep = np.array([int(h[6]) for h in hdr])

    classes = MSTAR_CLASSES if args.classes == "all" else PAPER_CLASSES
    print(f"本次转换 {len(classes)} 类：{', '.join(classes.values())}")

    # 按类组织：class → (indices, angles)
    by_class = {}
    for key in classes:
        m = (ttype == key) & (dep == args.depression)
        if m.sum() == 0:
            print(f"[warn] {classes[key]}({key}) 在俯角 {args.depression}° 无样本，"
                  f"尝试其它俯角...")
            m = ttype == key  # 回退到该类别所有俯角
        if m.sum() == 0:
            raise SystemExit(f"[error] 类别 {key} 完全没有样本")
        idx = np.where(m)[0]
        by_class[key] = (idx, az[idx])

    # 每类选 72 个网格角度
    chips = {}   # key -> {angle: (chip128, ...)}
    print("\n=== 各类网格角度匹配 ===")
    for key, label in classes.items():
        idx, ang = by_class[key]
        sel_ang, sel_idx = select_grid(ang, args.grid)
        sel_idx = idx[sel_idx]  # 映射回全局索引
        d = np.abs((sel_ang - np.arange(0, 360, args.grid)[:len(sel_ang)] + 180) % 360 - 180)
        print(f"  {label:>8}: {len(sel_ang)}/{int(360/args.grid)} 个角度, "
              f"最近邻偏差 mean={d.mean():.2f}° max={d.max():.2f}°")
        chips[key] = {round(float(a), 1): resize_128(mag[i]) for a, i in zip(sel_ang, sel_idx)}

    # 训练集（360 张）+ 评估集
    train_real, train_angles = [], []
    for key in classes:
        for a in sorted(chips[key]):
            train_real.append(chips[key][a])
            train_angles.append(a)
    train_real = np.stack(train_real)
    train_angles = np.asarray(train_angles, dtype=np.float64)

    # 评估集：每类挑 eval-per-class 个角度，含边界（0/5/350/355 之类）用于循环距离
    eval_ang, eval_class = [], []
    border = [0.0, 5.0, 90.0, 180.0, 270.0, 350.0, 355.0]
    for key in classes:
        avail = sorted(chips[key])
        # 边界优先 + 随机补足
        picked = [a for a in border if a in avail]
        rest = [a for a in avail if a not in picked]
        rng = np.random.RandomState(0)
        need = args.eval_per_class - len(picked)
        if need > 0:
            picked += rng.choice(rest, size=min(need, len(rest)), replace=False).tolist()
        for a in picked:
            eval_ang.append(a)
            eval_class.append(key)
    eval_ang = np.asarray(eval_ang, dtype=np.float64)

    real, real2, fake = [], [], []
    for k, (a, key) in enumerate(zip(eval_ang, eval_class)):
        c = chips[key][a]
        real.append(c)
        real2.append(second_look(c, seed=5000 + k))
        # fake：同类别、真实角度 = a + offset 的切片，再退化（含 wrap）
        a_fake = (a + args.angle_offset) % 360.0
        a_fake = min(chips[key], key=lambda x: abs((x - a_fake + 180) % 360 - 180))
        fake.append(degrade_real(chips[key][a_fake], blur_sigma=args.blur_sigma,
                                 noise_frac=args.noise_frac, looks_fake=args.looks_fake,
                                 seed=6000 + k))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "train_real.npy"), train_real)
    np.save(os.path.join(args.out, "train_angles.npy"), train_angles)
    for k in range(len(eval_ang)):
        np.save(os.path.join(args.out, f"real_{k:03d}.npy"), real[k])
        np.save(os.path.join(args.out, f"real2_{k:03d}.npy"), real2[k])
        np.save(os.path.join(args.out, f"fake_{k:03d}.npy"), fake[k])
    with open(os.path.join(args.out, "angles.json"), "w") as fh:
        json.dump({"angles": [float(a) for a in eval_ang],
                   "angle_offset": args.angle_offset}, fh)

    print(f"\n已生成 {args.out}/：train_real {train_real.shape}（{len(train_real)} 张 128×128）"
          f"、real/fake 各 {len(eval_ang)} 张（offset={args.angle_offset}°）。")
    print("幅度值域：min=%.3f max=%.3f" % (mag.min(), mag.max()))


if __name__ == "__main__":
    main()
