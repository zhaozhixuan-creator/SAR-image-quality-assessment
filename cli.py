#!/usr/bin/env python3
"""SAR 图像质检命令行入口。

用法：
    python cli.py <图像路径> [--metadata meta.json] [--domain auto|amplitude|intensity|db]
                  [--output report.html] [--json report.json]
                  [--nominal-resolution 3.0] [--pixel-spacing 10.0]

输入一张现成 SAR 图像（PNG/JPG/TIFF），输出 HTML 质检报告 + JSON 结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from sar_iqa.io import load_image, load_metadata
from sar_iqa.metrics import run_all
from sar_iqa.grading import grade
from sar_iqa.report import generate_html
from sar_iqa.profiles import _c0
from sar_iqa.modules.despeckling import (
    estimate_enl, lee_filter, ratio_image_test, detect_edges, epd_roa, m_index,
)
from sar_iqa.modules.metadata_rules import run_rules


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化: {type(o)}")


def run_despeckling_module(sar) -> dict:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SAR 图像质检：单图输入 → HTML 质检报告")
    ap.add_argument("image", help="输入图像路径（PNG/JPG/TIFF）")
    ap.add_argument("--metadata", help="元数据边车 JSON（用于元数据规则引擎与定标追溯）")
    ap.add_argument("--domain", default="auto", choices=["auto", "amplitude", "intensity", "db"])
    ap.add_argument("--channels", type=int, default=None, help="强制通道数解释（默认自动）")
    ap.add_argument("--output", help="HTML 报告输出路径（默认 <图名>_report.html）")
    ap.add_argument("--json", help="JSON 结果输出路径（默认 <图名>_report.json）")
    ap.add_argument("--nominal-resolution", type=float, default=None, help="标称分辨率(m)，用于展宽因子")
    ap.add_argument("--pixel-spacing", type=float, default=None, help="像素间距(m)，用于分辨率单位换算")
    ap.add_argument("--irf-window", type=int, default=64)
    ap.add_argument("--oversampling-factor", type=int, default=16)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"错误：找不到图像文件 {args.image}", file=sys.stderr)
        return 1

    # 元数据
    metadata = {}
    if args.metadata:
        metadata = load_metadata(args.metadata)
    if args.nominal_resolution:
        metadata["nominal_resolution"] = args.nominal_resolution
    if args.pixel_spacing:
        metadata["pixel_spacing"] = args.pixel_spacing

    # 加载
    print(f"[1/5] 读取图像 {args.image} ...")
    sar = load_image(args.image, domain=args.domain, channels=args.channels, metadata=metadata)
    print(f"      尺寸 {sar.W}×{sar.H}，通道 {sar.n_channels}，输入域 {sar.domain}")

    # 运行指标
    print("[2/5] 运行 7 维 38 项质检指标 ...")
    config = {"irf_window": args.irf_window, "oversampling_factor": args.oversampling_factor}
    results, ctx = run_all(sar, config)

    # 自研模块
    print("[3/5] 运行自研模块（去斑评价 / 元数据规则） ...")
    despeckling = run_despeckling_module(sar)
    rule_results = run_rules(metadata if metadata else None)

    # 分级
    print("[4/5] 缺陷分级与评分 ...")
    gr = grade(results)

    # 输出
    stem = os.path.splitext(args.image)[0]
    out_html = args.output or f"{stem}_report.html"
    out_json = args.json or f"{stem}_report.json"
    html_text = generate_html(results, ctx, gr, despeckling, rule_results)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    payload = {
        "image": args.image,
        "input": {"shape": [sar.H, sar.W], "channels": sar.n_channels,
                  "channel_names": sar.channel_names, "domain": sar.domain},
        "grade": gr,
        "despeckling": despeckling,
        "metrics": [r.as_dict() for r in results],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[5/5] 完成：评分 {gr['score']:.1f}（{gr['level']}）")
    print(f"      HTML 报告 → {out_html}")
    print(f"      JSON 结果 → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
