"""图像读写与 manifest 工具。

约定：所有指标面向「幅度域 [0,1]」灰度图（list[np.ndarray]，每张 (H,W) float32）。
真实参考与生成图统一转成该口径后再进入 v2 指标。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_gray_01(path) -> np.ndarray:
    """读图 → 单通道灰度幅度 [0,1] float32。RGB 取第一通道（v2 的 to_2d 口径一致）。"""
    im = Image.open(path)
    if im.mode not in ("L", "I", "F"):
        im = im.convert("L")
    a = np.asarray(im, dtype=np.float32)
    if a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def read_gray_u8(path) -> np.ndarray:
    """读图 → 单通道灰度 uint8（用于生成侧 / 报告侧可视化）。"""
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    return np.asarray(im, dtype=np.uint8)


def save_gray_01_png(path, arr: np.ndarray) -> None:
    """把 [0,1] 幅度灰度写为 PNG（缩放为 uint8）。"""
    u = np.rint(np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u, mode="L").save(path)


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def stack_01(images) -> np.ndarray:
    """list[(H,W)] → (N,H,W) float32。"""
    return np.stack([np.asarray(a, dtype=np.float32) for a in images], axis=0)
