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

from sar_iqa.single import assess_image, _json_default
from sar_iqa.report import generate_html


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
    ap.add_argument("--level", default="auto",
                    help="产品级别：auto/L0/L1(SLC)/L2/L3+（默认 auto→L1/SLC）")
    ap.add_argument("--batch", type=int, default=None,
                    help="批量大小，用于 GB/T 24356 抽样建议（≥1001 分批）")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"错误：找不到图像文件 {args.image}", file=sys.stderr)
        return 1

    metadata_extra = {}
    if args.nominal_resolution:
        metadata_extra["nominal_resolution"] = args.nominal_resolution
    if args.pixel_spacing:
        metadata_extra["pixel_spacing"] = args.pixel_spacing

    print(f"[1/5] 读取图像 {args.image} ...")
    config = {"irf_window": args.irf_window, "oversampling_factor": args.oversampling_factor}
    a = assess_image(args.image, domain=args.domain, channels=args.channels,
                     metadata_path=args.metadata, metadata_extra=metadata_extra,
                     level=args.level, config=config, batch=args.batch)
    print(f"      尺寸 {a.sar.W}×{a.sar.H}，通道 {a.sar.n_channels}，输入域 {a.sar.domain}")

    print("[2/5] 运行 7 维 38 项质检指标（分级流水线） ...")
    print("[3/5] 运行自研模块（去斑评价 / 元数据规则） ...")
    print("[4/5] 缺陷分级与评分 ...")

    stem = os.path.splitext(args.image)[0]
    out_html = args.output or f"{stem}_report.html"
    out_json = args.json or f"{stem}_report.json"
    html_text = generate_html(a.results, a.ctx, a.grade, a.despeckling, a.rule_results,
                              pipeline=a.pipeline, sample=a.sample)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(a.payload(), f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[5/5] 完成：评分 {a.grade['score']:.1f}（{a.grade['level']}），产品级 {a.pipeline.level}")
    print(f"      HTML 报告 → {out_html}")
    print(f"      JSON 结果 → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
