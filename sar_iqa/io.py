"""图像读取与归一化：把一张现成 SAR 图像读进统一的数据容器。

支持 PNG / JPG / TIFF（8bit / 16bit / float）。自动识别单通道 / 双极化 / 四极化。
数据统一归一到三个等价表示：intensity（线性功率）、amplitude（幅度）、db（分贝）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

EPS = 1e-12

# 四极化通道命名（SLC 标准顺序）
QUAD_NAMES = ["HH", "HV", "VH", "VV"]
DUAL_NAMES = ["co-pol", "cross-pol"]


@dataclass
class SarImage:
    """统一的数据容器，所有指标从这里取数。"""

    path: str
    data: np.ndarray            # 原始数值 float64，(H, W) 或 (H, W, C)
    n_channels: int
    channel_names: list[str]
    domain: str                 # 输入域：amplitude / intensity / db
    intensity: np.ndarray       # 线性功率
    amplitude: np.ndarray       # 幅度
    db: np.ndarray              # 10*log10(intensity)
    metadata: dict = field(default_factory=dict)
    is_polarimetric: bool = False

    @property
    def shape(self):
        return self.data.shape

    @property
    def H(self):
        return self.data.shape[0]

    @property
    def W(self):
        return self.data.shape[1]


def _to_intensity(x: np.ndarray, domain: str) -> np.ndarray:
    """把输入域转成线性功率（intensity）。"""
    x = np.asarray(x, dtype=np.float64)
    if domain in ("amplitude", "magnitude"):
        return x ** 2
    if domain == "intensity":
        return np.maximum(x, 0.0)
    if domain == "db":
        return 10 ** (x / 10.0)
    raise ValueError(f"未知输入域: {domain}")


def _read_array(path: str) -> np.ndarray:
    """读取图像为 float64 ndarray。

    优先用 PIL（支持 PNG/JPG/TIFF 常见格式）；PIL 无法识别的格式
    （如 tifffile 写入的多通道 float32 TIFF）回退到 tifffile 读取。
    """
    try:
        img = Image.open(path)
        return np.asarray(img, dtype=np.float64)
    except Exception:
        try:
            import tifffile
        except ImportError as e:  # pragma: no cover - 依赖缺失时给出明确提示
            raise RuntimeError(
                f"无法读取 {path}：PIL 不支持该格式，且未安装 tifffile。"
            ) from e
        return np.asarray(tifffile.imread(path), dtype=np.float64)


def load_image(path: str, domain: str = "auto", channels: Optional[int] = None,
               metadata: Optional[dict] = None) -> SarImage:
    """读取并归一化一张图像。

    domain: auto / amplitude / intensity / db
    channels: 强制指定通道数解释（默认自动）
    metadata: 可选元数据字典（用于元数据规则引擎与定标追溯）
    """
    arr = _read_array(path)

    # 判定通道
    if arr.ndim == 2:
        arr = arr[:, :, None]
    c = arr.shape[2]

    if channels is not None:
        n_ch = channels
        # 若指定单通道但给了 3 通道图，取亮度
        if n_ch == 1 and c == 3:
            arr = _rgb2gray(arr)[:, :, None]
            c = 1
        elif n_ch == 1 and c > 1:
            arr = arr[:, :, 0:1]
            c = 1
        else:
            n_ch = c
    else:
        if c == 3:
            # 3 通道按 RGB 可视化图处理，转灰度亮度
            arr = _rgb2gray(arr)[:, :, None]
            c = 1
        n_ch = c

    # 通道命名
    if n_ch == 1:
        names = ["Intensity"]
        is_pol = False
    elif n_ch == 2:
        names = list(DUAL_NAMES)
        is_pol = True
    elif n_ch == 4:
        names = list(QUAD_NAMES)
        is_pol = True
    else:
        names = [f"ch{i}" for i in range(n_ch)]
        is_pol = n_ch >= 2

    # 输入域
    if domain == "auto":
        neg_frac = float((arr < 0).mean())
        if neg_frac > 0.05:
            dom = "db"            # 大范围负值 → 大概率是 dB
        else:
            dom = "amplitude"     # 少量负值视为噪声减除伪影，仍按幅度处理
    else:
        dom = domain

    intensity = _to_intensity(arr, dom)
    amplitude = np.sqrt(intensity)
    db = 10.0 * np.log10(np.maximum(intensity, EPS))

    return SarImage(
        path=path,
        data=arr,
        n_channels=n_ch,
        channel_names=names,
        domain=dom,
        intensity=intensity,
        amplitude=amplitude,
        db=db,
        metadata=metadata or {},
        is_polarimetric=is_pol,
    )


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])[:, :, None]


def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
