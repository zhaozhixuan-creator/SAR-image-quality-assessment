#!/usr/bin/env python3
"""SAR 图像数据集级质检命令行入口。

用法：
    python dataset_cli.py <目录> [--level auto|L0|L1|L2|L3+]
                         [--recursive] [--ext tif,tiff,png,jpg,jpeg]
                         [--out-dir DIR] [--no-per-image]
                         [--group-by processor_version] [--mad-k 3.0]

输入一个 SAR 图像目录（每个图像可选同茎 .json 元数据边车），
输出数据集级看板（dashboard.html）+ 汇总 JSON（dataset_summary.json）
+ 逐图落库（records.json）+ 每图报告（per_image/<stem>.html/.json）。
"""
from __future__ import annotations

import argparse
import os
import sys

from sar_iqa.dataset import run_dataset, IMAGE_EXTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SAR 数据集质检：目录扫描 → 数据集级看板 + 逐图报告")
    ap.add_argument("dir", help="输入图像目录（每个图像可选同茎 .json 元数据边车）")
    ap.add_argument("--level", default="auto",
                    help="产品级别：auto/L0/L1(SLC)/L2/L3+（默认 auto→L1/SLC）")
    ap.add_argument("--domain", default="auto", choices=["auto", "amplitude", "intensity", "db"])
    ap.add_argument("--channels", type=int, default=None, help="强制通道数解释（默认自动）")
    ap.add_argument("--nominal-resolution", type=float, default=None, help="标称分辨率(m)，覆盖全部图像")
    ap.add_argument("--pixel-spacing", type=float, default=None, help="像素间距(m)，覆盖全部图像")
    ap.add_argument("--irf-window", type=int, default=64)
    ap.add_argument("--oversampling-factor", type=int, default=16)
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    ap.add_argument("--ext", default=",".join(IMAGE_EXTS).replace(".", ""),
                    help="图像扩展名（逗号分隔，不含点）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 <目录>_qc）")
    ap.add_argument("--no-per-image", action="store_true", help="不生成 per_image/ 单图报告")
    ap.add_argument("--group-by", default=None,
                    help="分组审计键（元数据字段，如 processor_version / sensor；默认按产品级）")
    ap.add_argument("--batch", type=int, default=None, help="覆盖抽样批量（默认 = 图像数）")
    ap.add_argument("--mad-k", type=float, default=3.0, help="离群检测 k 值（默认 3.0）")
    ap.add_argument("--metadata-keys", default=None,
                    help="元数据一致性字段（逗号分隔；默认 processor_version 等身份键）")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"错误：找不到目录 {args.dir}", file=sys.stderr)
        return 1

    exts = tuple("." + e.strip().lstrip(".") for e in args.ext.split(",") if e.strip())
    config = {"irf_window": args.irf_window, "oversampling_factor": args.oversampling_factor}

    metadata_extra = {}
    if args.nominal_resolution:
        metadata_extra["nominal_resolution"] = args.nominal_resolution
    if args.pixel_spacing:
        metadata_extra["pixel_spacing"] = args.pixel_spacing

    metadata_keys = None
    if args.metadata_keys:
        metadata_keys = tuple(k.strip() for k in args.metadata_keys.split(",") if k.strip())

    print(f"[1/3] 扫描目录 {args.dir} ...")
    result = run_dataset(
        args.dir, level=args.level, domain=args.domain, channels=args.channels,
        config=config, metadata_extra=metadata_extra, out_dir=args.out_dir,
        per_image=not args.no_per_image, recursive=args.recursive,
        group_by=args.group_by, batch_size=args.batch, mad_k=args.mad_k,
        metadata_keys=metadata_keys, exts=exts,
    )

    summary = result.as_dict()
    mean = result.batch.get("score_mean")
    mean_txt = f"{mean:.1f}" if mean is not None else "—"
    print(f"[2/3] 聚合统计 / 批次判定 / 离群检测 / 分组审计 ...")
    print(f"[3/3] 完成：")
    print(f"      图像 {summary['n_images_total']} 张"
          f"（成功 {summary['n_images_ok']}，失败 {summary['n_images_error']}）")
    print(f"      整批结论：{result.batch['verdict']}（平均分 {mean_txt}）")
    print(f"      看板 → {os.path.join(result.root, 'dashboard.html')}")
    print(f"      汇总 → {os.path.join(result.root, 'dataset_summary.json')}")
    print(f"      落库 → {os.path.join(result.root, 'records.json')}")
    if not args.no_per_image:
        print(f"      每图报告 → {os.path.join(result.root, 'per_image')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
