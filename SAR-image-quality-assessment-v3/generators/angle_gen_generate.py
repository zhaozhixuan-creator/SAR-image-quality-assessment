#!/usr/bin/env python3
"""angle_gen（XUNet 扩散 NVS）生成封装。

在 angle_gen/.venv 内、cwd=angle_gen 目录运行。复用 NVSynthesis 的配置解析与
模型搭建，仅重写测试循环，把每对 (参考, 目标, 生成) 原始灰度图 + 角度逐张落盘，
供 v3 后续 FR 配对与指标计算。

用法：
    python generators/angle_gen_generate.py --config <test.yml> --outdir <dir> \
        --per-class 20 --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# angle_gen 根目录（依赖 configs / NVSynthesis / data / models 包）
REPO_ROOT = Path(os.getcwd()).resolve()
for p in (str(REPO_ROOT), str(REPO_ROOT / "NVSynthesis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from NVSynthesis.apis.main import NVSynthesis  # noqa: E402


def _tensor_to_01(t) -> np.ndarray:
    """[-1,1] 张量 → [0,1] 灰度 (H,W) float32。"""
    t = t.detach().float().cpu().squeeze()
    if t.dim() == 3:
        t = t[0] if t.size(0) == 1 else t.mean(0)
    a = (t.clamp(-1, 1) + 1.0) / 2.0
    return a.numpy().astype(np.float32)


def _angle_deg(v) -> float:
    if torch.is_tensor(v):
        v = v.detach().cpu().item()
    return float(v) * 180.0 / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test-target-type", default="all")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[angle_gen] init NVSynthesis from {args.config}")
    tester = NVSynthesis(is_train=False, opt=args.config, test_target_type=args.test_target_type)

    # 复用 main.test() 的设备与 record 逻辑
    pipL = tester.pipL
    device = getattr(tester.scheme, "device", torch.device(args.device))

    manifest = []
    per_class_done = {}

    for test_loader, dataset_opt in zip(tester.test_loader_list, tester.test_dataset_opt):
        index_start = list(dataset_opt.get("index_start", []))
        for idx_data, test_data in enumerate(test_loader):
            img_name = test_data.get("img_name", "unknown")
            if isinstance(img_name, (list, tuple)):
                img_name = str(img_name[0])
            target_type = str(img_name)

            # 每个类的首样本仅用于设置参考 condition，不生成
            if idx_data in index_start:
                pipL.record = [[
                    test_data["img"][:, 0],
                    test_data["azimuth_angle"][:, 0].unsqueeze(1),
                    test_data["incidence_angle"][:, 0].unsqueeze(1),
                ]]
                continue

            if per_class_done.get(target_type, 0) >= args.per_class:
                continue

            output_dict = pipL.Test_Pipeline(**test_data)
            pred_lq = output_dict["z"]
            min_w_idx = int(torch.argmin(pipL.w).item())
            generated = pred_lq[min_w_idx].detach().cpu()

            ref = test_data["img"][0, 0].detach().cpu()
            gt = test_data["img"][0, 1].detach().cpu()
            src_az = _angle_deg(test_data["azimuth_angle"][0, 0])
            tgt_az = _angle_deg(test_data["azimuth_angle"][0, 1])
            src_inc = _angle_deg(test_data["incidence_angle"][0, 0])
            tgt_inc = _angle_deg(test_data["incidence_angle"][0, 1])

            k = per_class_done.get(target_type, 0)
            fid = f"{target_type}_{k:04d}"
            np.save(outdir / f"{fid}_ref.npy", _tensor_to_01(ref))
            np.save(outdir / f"{fid}_gt.npy", _tensor_to_01(gt))
            np.save(outdir / f"{fid}_gen.npy", _tensor_to_01(generated))
            manifest.append({
                "target_type": target_type,
                "ref_idx": fid,
                "src_azimuth_deg": src_az,
                "src_incidence_deg": src_inc,
                "tgt_azimuth_deg": tgt_az,
                "tgt_incidence_deg": tgt_inc,
                "ref_file": f"{fid}_ref.npy",
                "gt_file": f"{fid}_gt.npy",
                "gen_file": f"{fid}_gen.npy",
            })
            per_class_done[target_type] = k + 1

    (outdir / "generation_manifest.json").write_text(
        json.dumps({"total": len(manifest), "samples": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[angle_gen] done: {len(manifest)} samples -> {outdir}")


if __name__ == "__main__":
    main()
