"""Stage A —— 数据准备：把真实 MSTAR 参考集归一化为幅度域 [0,1] 灰度图（128/64 两档），
并附每样本 (类别, 方位角, 入射角) 元数据，供后续配对与指标计算。

输入：
  - config.datasets.image_root         真实 .jpg（train/test 按类分子目录）
  - config.datasets.metadata_{train,test}  meta.pkl（类名 → [样本 dict]，含 image_path /
                                          azimuth_angle(弧度) / target_azimuth_deg /
                                          incidence_angle(弧度) / measured_depression_deg）
输出（workspace/real/{size}/{split}/）：
  - images.npy    (N, size, size) float32 [0,1]
  - labels.json   [{class_idx, class_name, azimuth_deg, incidence_deg, image_path}]
  - summary.json  统计（各类数量、角度范围）
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import io, paths


def _fit_gray_01(path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.fit(im.convert("L"), (size, size),
                          method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        a = np.asarray(im, dtype=np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)


def _load_meta(meta_path, cfg) -> dict:
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    return meta


def build_split(cfg, split: str, size: int) -> dict:
    class_names = cfg["datasets"]["class_names"]
    meta_path = paths.resolve(cfg["datasets"][f"metadata_{split}"], cfg)
    meta = _load_meta(meta_path, cfg)

    images, labels = [], []
    class_counts = {c: 0 for c in class_names}
    azimuths, incidences = [], []

    for ci, cls in enumerate(class_names):
        rows = sorted(meta[cls], key=lambda r: (float(r.get("target_azimuth_deg", 0.0)), r["image_path"]))
        for row in rows:
            src = paths.remap_remote(row["image_path"], cfg)
            images.append(_fit_gray_01(src, size))

            az_deg = float(row["target_azimuth_deg"])
            # incidence：meta 里 incidence_angle 为弧度；同时保留 depression 换算口径
            inc_rad = float(row.get("incidence_angle", 0.0))
            inc_deg = math.degrees(inc_rad) if inc_rad else float(90.0 - float(row["measured_depression_deg"]))
            labels.append({
                "class_idx": ci,
                "class_name": cls,
                "azimuth_deg": az_deg,
                "incidence_deg": inc_deg,
                "image_path": src,
            })
            class_counts[cls] += 1
            azimuths.append(az_deg)
            incidences.append(inc_deg)

    arr = np.stack(images, axis=0).astype(np.float32)
    return {
        "images": arr,
        "labels": labels,
        "class_counts": class_counts,
        "azimuth_deg_range": [float(min(azimuths)), float(max(azimuths))],
        "incidence_deg_range": [float(min(incidences)), float(max(incidences))],
    }


def run(cfg, args) -> None:
    sizes = cfg["datasets"]["sizes"]
    for size in sizes:
        for split in ["train", "test"]:
            out = paths.ws_dir(cfg, "real", str(size), split)
            out.mkdir(parents=True, exist_ok=True)
            data = build_split(cfg, split, size)
            np.save(out / "images.npy", data["images"])
            io.write_json(out / "labels.json", data["labels"])
            summary = {
                "split": split, "size": size, "sample_count": int(data["images"].shape[0]),
                "class_counts": data["class_counts"],
                "azimuth_deg_range": data["azimuth_deg_range"],
                "incidence_deg_range": data["incidence_deg_range"],
            }
            io.write_json(out / "summary.json", summary)
            print(f"[stage_a] real/{size}/{split}: {data['images'].shape[0]} 张, "
                  f"方位角 [{summary['azimuth_deg_range'][0]:.1f}, {summary['azimuth_deg_range'][1]:.1f}]°")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    cfg = paths.load_config()
    run(cfg, None)
