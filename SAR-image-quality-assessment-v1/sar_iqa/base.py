"""基础数据结构：指标结果、状态枚举、注册表。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class Status(str, Enum):
    """单项指标判定的四态。

    - PASS   : 合格（有数值，落在判据内）
    - WARN   : 需关注（有数值，但为代理值 / 越界轻微 / 数据不完整）
    - FAIL   : 超差 / 检出缺陷
    - NODATA : 无法评估（单张图像拿不到所需数据，需角反射器 / 时序 / 干涉对 / 原始回波）
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NODATA = "nodata"


# 状态在报告中显示的元信息（标签 + 颜色 + 说明）
STATUS_META = {
    Status.PASS: {"label": "合格", "color": "#2e7d32", "bg": "#e8f5e9"},
    Status.WARN: {"label": "需关注", "color": "#ef6c00", "bg": "#fff3e0"},
    Status.FAIL: {"label": "超差", "color": "#c62828", "bg": "#ffebee"},
    Status.NODATA: {"label": "无法评估", "color": "#546e7a", "bg": "#eceff1"},
}


@dataclass
class MetricResult:
    """单个指标的测量结果。"""

    key: str                       # 稳定英文标识，如 "radiation.saturation"
    name: str                      # 中文名，如 "饱和率 / 动态范围"
    dimension: str                 # 所属维度，如 "辐射质量"
    value: Optional[float] = None  # 主数值
    unit: str = ""                 # 单位
    status: Status = Status.NODATA
    reason: str = ""               # 判定理由 / 无法评估原因
    threshold: str = ""            # 判据 / 基准文本
    detail: dict = field(default_factory=dict)  # 附带的次级数据
    raw: Any = None                # 原始对象（剖面、掩膜等，供报告绘图）

    # 规格元信息（由 spec.annotate 按 key 回填，非各维度模块填写）
    level: str = ""                # 原生产品级别：L0 / L1/SLC / L2 / L3+
    phase: int = 0                 # 落地分期：1 / 2 / 3
    kind: str = "decision"         # decision（判决项）/ marker（标记项）
    method: str = ""               # 测量方法
    refs: list = field(default_factory=list)  # 参考开源实现（工具键）

    def as_dict(self) -> dict:
        d = {
            "key": self.key,
            "name": self.name,
            "dimension": self.dimension,
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "reason": self.reason,
            "threshold": self.threshold,
            "detail": {k: _jsonable(v) for k, v in self.detail.items()},
        }
        if self.level:
            d["level"] = self.level
        if self.phase:
            d["phase"] = self.phase
        if self.kind:
            d["kind"] = self.kind
        if self.refs:
            d["refs"] = self.refs
        return d


def _jsonable(v):
    """把 detail 里的 numpy 标量转成 python 原生类型。"""
    if hasattr(v, "item"):
        return v.item()
    return v


# 指标注册表：key -> 计算函数。各维度模块启动时自行注册。
REGISTRY: dict[str, Callable] = {}


def register(key: str):
    def deco(fn):
        REGISTRY[key] = fn
        return fn
    return deco
