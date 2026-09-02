#!/usr/bin/env python3
"""从 mstar_raw 展开完整 MSTAR 参考集（10 类 × 72 角 = 720 张，128×128 幅度）。

输出到 mstar_ref/，按类组织，供 run1_float 生成图的 FR 指标「同类交叉配对」使用。

类索引 0–9 与 run1_float/labels.npy 完全一致：
  0:2S1  1:BMP2  2:BRDM2  3:BTR60  4:BTR70  5:D7  6:T62  7:T72  8:ZIL131  9:ZSU234

与 mstar_to_chips.py 的关系：本脚本只产出「干净的按类参考集」，不生成
real/real2/fake 评估集，也不覆盖 mstar_real/，与现有评估口径互不干扰。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 类名 → 显示名（dict 顺序即类索引 0–9，与 run1_float 对齐）
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
    "zsu23-4_gun": "ZSU234",
}


def load_all(raw_dir):
    hdr, mag = [], []
    for name in ["SAR10a.npz", "SAR10b.npz", "SAR10c.npz"]:
        p = os.path.join(raw_dir, name)
        d = np.load(p, allow_pickle=True)
        hdr.append(d["hdr"])
        mag.append(d["mag"])
    return np.concatenate(hdr, axis=0), np.concatenate(mag, axis=0)


def select_grid(angles, grid_step=5.0):
    """为每个 5° 网格点选一个最近样本（去重），返回 (选中角度, 选中原始索引)。"""
    grid = np.arange(0, 360, grid_step)
    used = np.zeros(len(angles), dtype=bool)
    sel_ang, sel_idx = [], []
    for g in grid:
        d = np.abs((angles - g + 180.0) % 360.0 - 180.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_raw"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_ref"))
    ap.add_argument("--depression", type=int, default=17, help="固定俯角（17=SOC）")
    ap.add_argument("--grid", type=float, default=5.0, help="方位角网格步长（度）")
    args = ap.parse_args()

    hdr, mag = load_all(args.raw)
    ttype = np.array([str(h[0]).strip().lower() for h in hdr])
    az = np.array([float(h[2]) for h in hdr])
    dep = np.array([int(h[6]) for h in hdr])
    print(f"读入 {len(hdr)} 张 MSTAR 切片（原始 {mag.shape[1]}×{mag.shape[2]}）")

    imgs, labels, angles = [], [], []
    class_names = list(MSTAR_CLASSES.values())
    for ci, (key, label) in enumerate(MSTAR_CLASSES.items()):
        m = (ttype == key) & (dep == args.depression)
        if m.sum() == 0:
            print(f"[warn] {label}({key}) 在俯角 {args.depression}° 无样本，回退全部俯角")
            m = ttype == key
        idx = np.where(m)[0]
        sel_ang, sel_idx = select_grid(az[idx], args.grid)
        sel_idx = idx[sel_idx]
        for a, i in zip(sel_ang, sel_idx):
            imgs.append(resize_128(mag[i]))
            labels.append(ci)
            angles.append(float(a))
        print(f"  {label:>8}: {len(sel_ang)} 个角度")

    imgs = np.stack(imgs).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    angles = np.asarray(angles, dtype=np.float64)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "ref_images.npy"), imgs)
    np.save(os.path.join(args.out, "ref_labels.npy"), labels)
    np.save(os.path.join(args.out, "ref_angles.npy"), angles)
    with open(os.path.join(args.out, "class_names.json"), "w") as fh:
        json.dump(class_names, fh, ensure_ascii=False, indent=2)

    print(f"\n已生成 {args.out}/：ref_images {imgs.shape}，ref_labels {labels.shape}，"
          f"ref_angles {angles.shape}")
    print("幅度值域：min=%.3f max=%.3f" % (imgs.min(), imgs.max()))
    per = {c: int((labels == i).sum()) for i, c in enumerate(class_names)}
    print("每类数量：", per)


if __name__ == "__main__":
    main()
