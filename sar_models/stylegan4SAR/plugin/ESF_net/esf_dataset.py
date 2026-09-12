import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _canonical_sort(points: np.ndarray) -> np.ndarray:
    # 使用稳定排序把点集变成“可回归”的固定顺序:
    # 先按 x，再按 y；避免每个样本点序随机导致网络学不稳。
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


class ASCZipDataset(Dataset):
    """Image -> ASC attribute points dataset.

    Input: grayscale image [1,128,128]
    Target: [K,4] = [x_norm, y_norm, amp_A, alpha]
    """

    def __init__(self, zip_path: Path, k_points: int, attr_dim: int) -> None:
        self.zip_path = Path(zip_path)
        self.k_points = int(k_points)
        self.attr_dim = int(attr_dim)
        self._zip = zipfile.ZipFile(self.zip_path, "r")

        with self._zip.open("labels.jsonl", "r") as f:
            labels = [json.loads(line.decode("utf-8")) for line in f]

        self.rows: List[Dict] = []
        for row in labels:
            pts = row["asc"]["points_padded"]
            if len(pts) != self.k_points:
                continue
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def _build_target(self, points: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = np.zeros((self.k_points, self.attr_dim), dtype=np.float32)
        msk = np.zeros((self.k_points, 1), dtype=np.float32)
        valid_pts = []
        for p in points:
            if int(p.get("valid", 0)) != 1:
                continue
            valid_pts.append([float(p["x_norm"]), float(p["y_norm"]), float(p["amp_A"]), float(p["alpha"])])

        if len(valid_pts) > 0:
            arr = np.asarray(valid_pts, dtype=np.float32)
            arr = _canonical_sort(arr)
            n = min(arr.shape[0], self.k_points)
            tgt[:n, : self.attr_dim] = arr[:n, : self.attr_dim]
            msk[:n, 0] = 1.0
        return torch.from_numpy(tgt), torch.from_numpy(msk)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_name = row["image"]
        with self._zip.open(image_name, "r") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw)).convert("L")
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0)

        target, mask = self._build_target(row["asc"]["points_padded"])
        meta = {
            "image_name": image_name,
            "sample_id": row.get("sample_id", ""),
            "class_name": row.get("meta", {}).get("class_name", ""),
        }
        return x, target, mask, meta
