"""单图评估单元：把一次单图质检封为一个可复用函数，供两条入口共用。

`cli.py`（单图入口）与 `dataset.py`（数据集入口）都调用 `assess_image`，
避免重复单图评估流程。`run_despeckling_module` 与 `_json_default` 原在
`cli.py` 局部定义，迁移到此供两条入口共用，`cli.py` 行为保持不变。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import SarImage, load_image, load_metadata
from .pipeline import PipelineResult, run_pipeline
from .grading import grade, sample_size
from .profiles import _c0
from . import ecosystem
from .metrics import Context
from .modules.despeckling import (
    estimate_enl, lee_filter, ratio_image_test, detect_edges, epd_roa, m_index,
)
from .modules.metadata_rules import RuleResult, run_rules, summarize


def _json_default(o):
    """JSON 序列化兜底：numpy 标量 / 数组 → 原生类型。"""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化: {type(o)}")


def run_despeckling_module(sar: SarImage) -> dict:
    """自研模块 1：去斑评价指标包（内部 Lee 滤波作参考去斑）。"""
    I = _c0(sar.intensity)
    est = estimate_enl(sar.intensity)
    den = lee_filter(I)
    ratio = ratio_image_test(I, den)
    edges = detect_edges(I)
    epd = epd_roa(I, den, edges)
    m = m_index(I, den)
    return {
        "enl": {"enl_moment": est["enl_moment"], "enl_logcumulant": est["enl_logcumulant"],
                "diverged": est["diverged"], "n_uniform_pixels": est["n_pixels"]},
        "ratio": {"mean_bias": ratio["mean_bias"], "var_bias": ratio["var_bias"],
                  "spatial_ac": ratio["spatial_ac"]},
        "epd": {"epd": epd["epd"], "n_edges": epd["n_edges"]},
        "m_index": m,
    }


def _rule_to_dict(r: RuleResult) -> dict:
    return {"name": r.name, "clause": r.clause, "level": r.level,
            "passed": r.passed, "achieved": r.achieved, "reason": r.reason}


@dataclass
class Assessment:
    """一次单图质检的完整结果（含活动对象，供报告绘图）。"""

    image_path: str
    stem: str
    sar: SarImage
    metadata: dict
    results: list
    ctx: Context
    pipeline: PipelineResult
    grade: dict
    despeckling: dict | None
    rule_results: list
    rule_summary: dict
    sample: dict | None

    def payload(self) -> dict:
        """与 cli.py 单图 JSON 一致的结果，另附 rule_results（超集）。"""
        return {
            "image": self.image_path,
            "input": {"shape": [self.sar.H, self.sar.W], "channels": self.sar.n_channels,
                      "channel_names": self.sar.channel_names, "domain": self.sar.domain},
            "grade": self.grade,
            "pipeline": self.pipeline.as_dict(),
            "sampling": self.sample,
            "ecosystem": ecosystem.compliance_summary(),
            "despeckling": self.despeckling,
            "rule_results": [_rule_to_dict(r) for r in self.rule_results],
            "metrics": [r.as_dict() for r in self.results],
        }


def assess_image(image_path: str, *, domain: str = "auto", channels: Optional[int] = None,
                 metadata_path: Optional[str] = None, metadata_extra: Optional[dict] = None,
                 level: Optional[str] = "auto", config: Optional[dict] = None,
                 batch: Optional[int] = None) -> Assessment:
    """读图并完成一次单图质检，返回 Assessment（含活动 sar/ctx 供报告绘图）。

    metadata_extra 在边车加载后合并（如 --nominal-resolution / --pixel-spacing 注入）。
    """
    metadata: dict = {}
    if metadata_path:
        metadata = load_metadata(metadata_path)
    if metadata_extra:
        metadata.update(metadata_extra)

    sar = load_image(image_path, domain=domain, channels=channels, metadata=metadata)
    config = config or {}
    pr = run_pipeline(sar, level=level, config=config)
    results, ctx = pr.results, pr.ctx

    despeckling = run_despeckling_module(sar)
    rule_results = run_rules(metadata if metadata else None)
    gr = grade(results)
    sample = sample_size(batch) if batch is not None else None

    stem = os.path.splitext(os.path.basename(image_path))[0]
    return Assessment(
        image_path=image_path, stem=stem, sar=sar, metadata=metadata,
        results=results, ctx=ctx, pipeline=pr, grade=gr, despeckling=despeckling,
        rule_results=rule_results, rule_summary=summarize(rule_results), sample=sample,
    )
