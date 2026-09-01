#!/usr/bin/env python3
"""核查从 GitHub (jwcalder/MSTAR-Active-Learning) 下载的 MSTAR .npz 数据。

打印每类的样本数、俯角分布、方位角范围，确认 MSTAR 全部 10 类
（2S1/BMP2/BRDM2/BTR60/BTR70/D7/T62/T72/ZIL131/ZSU23-4）是否齐全、幅度值范围是否合理。
运行前需先下载 Data/SAR10{a,b,c}.npz 到 mstar_raw/。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mstar_raw")


def load_all():
    hdr, mag, phase = [], [], []
    for name in ["SAR10a.npz", "SAR10b.npz", "SAR10c.npz"]:
        p = os.path.join(RAW, name)
        if not os.path.exists(p):
            print(f"[跳过] 未找到 {p}")
            continue
        d = np.load(p, allow_pickle=True)
        hdr.append(d["hdr"])
        mag.append(d["mag"])
        phase.append(d["phase"])
        print(f"[读入] {name}: hdr={d['hdr'].shape} mag={d['mag'].shape} "
              f"phase={d['phase'].shape} fields={list(d['fields'])}")
    hdr = np.concatenate(hdr, axis=0)
    mag = np.concatenate(mag, axis=0)
    phase = np.concatenate(phase, axis=0)
    return hdr, mag, phase


def main():
    hdr, mag, phase = load_all()
    n = len(hdr)
    print(f"\n总样本数: {n}")

    # 字段顺序（fields[3:] 后）：TargetType, TargetSerNum, TargetAz, TargetRoll,
    # TargetPitch, TargetYaw, DesiredDepression
    ttype = np.asarray([str(h[0]).strip() for h in hdr])
    dep = np.asarray([int(h[6]) for h in hdr])
    az = np.asarray([float(h[2]) for h in hdr])

    print("\n=== 各类样本数 ===")
    uniq, cnts = np.unique(ttype, return_counts=True)
    for u, c in sorted(zip(uniq, cnts), key=lambda x: -x[1]):
        print(f"  {u!r:<20} {c}")

    print("\n=== 俯角分布 ===")
    for d, c in sorted(zip(*np.unique(dep, return_counts=True))):
        print(f"  {d}°: {c}")

    print("\n=== 全部 10 类在各俯角下的角度覆盖 ===")
    all_classes = {
        "2s1_gun": "2S1", "bmp2_tank": "BMP2", "brdm2_truck": "BRDM2",
        "btr60_transport": "BTR60", "btr70_transport": "BTR70", "d7_bulldozer": "D7",
        "t62_tank": "T62", "t72_tank": "T72", "zil131_truck": "ZIL131",
        "zsu23-4_gun": "ZSU23-4",
    }
    for key, label in all_classes.items():
        mask = np.array([k.strip().lower() == key for k in ttype])
        for d in sorted(set(dep[mask])):
            sub = az[mask & (dep == d)]
            if len(sub):
                print(f"  {label:>8} ({key}) 俯角{d}°: {len(sub)} 样本, "
                      f"方位角 [{sub.min():.0f}°, {sub.max():.0f}°]")

    print("\n=== 幅度值范围（应 ≥0 的浮点，量级数百~数千）===")
    print(f"  mag: min={mag.min():.3f} max={mag.max():.3f} mean={mag.mean():.3f}")
    print(f"  phase: min={phase.min():.3f} max={phase.max():.3f} mean={phase.mean():.3f}")


if __name__ == "__main__":
    main()
