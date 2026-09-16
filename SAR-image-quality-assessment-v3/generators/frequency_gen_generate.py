#!/usr/bin/env python3
"""frequency_gen（X→Ka Pix2Pix 翻译）生成封装。

在 frequency_gen/.venv 内、cwd=frequency_gen 目录运行。复用 mmgen 的配置解析与模型搭建，
把每一对「真实 Ka(gt) / 生成 Ka(fake)」六通道图降为单通道幅度 [0,1] 逐张落盘，
供 v3 后续 FR 配对与指标计算。

用法：
    python generators/frequency_gen_generate.py --config <pix2pix.py> \
        --checkpoint <latest.pth> --outdir <dir> --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# frequency_gen 仓库根目录（cwd 指向该仓库，含 vendored `mmgen` 包，test.sh 用 PYTHONPATH=$PWD 导入）
REPO_ROOT = os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model


def _channels_to_01(t) -> np.ndarray:
    """[B,6,H,W] 归一化 [-1,1] 张量 → 单通道 (H,W) 幅度 [0,1]（跨 6 通道取均值）。

    与 v2 指标数据契约一致：real/fake 均为 [0,1] 灰度图；六通道 PolSAR 用跨通道均值
    作为单通道代表幅度，real 与 fake 走同一降维，保证逐对可比。
    """
    t = t.detach().float().cpu()
    if t.dim() == 4:
        t = t.mean(dim=1)          # (B,H,W)
    t = t.squeeze(0)               # (H,W)
    a = (t.clamp(-1, 1) + 1.0) / 2.0
    return a.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dataroot", default=None,
                    help="覆盖测试数据根目录；在其下 testdir 子目录扫描配对 tiff（默认用配置里的相对路径）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.dataroot:
        cfg.data.test.dataroot = args.dataroot
    set_random_seed(2021, deterministic=False)

    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.eval()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[args.gpu])

    target_domain = model.module._default_domain              # 'ka'
    source_domain = model.module.get_other_domains(target_domain)[0]  # 'x'

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.get("val_workers_per_gpu", cfg.data.workers_per_gpu),
        dist=False,
        shuffle=False)

    manifest = []
    idx = 0
    for data_batch in data_loader:
        with torch.no_grad():
            output = model(data_batch[f"img_{source_domain}"],
                           test_mode=True, target_domain=target_domain)
        fake = output["target"]                          # [B,6,H,W] 生成 Ka
        real = data_batch[f"img_{target_domain}"]        # [B,6,H,W] 真值 Ka
        b = int(fake.size(0))
        for j in range(b):
            gt = _channels_to_01(real[j:j + 1])
            gen = _channels_to_01(fake[j:j + 1])
            fid = f"{idx:04d}"
            np.save(outdir / f"{fid}_gt.npy", gt)
            np.save(outdir / f"{fid}_gen.npy", gen)
            manifest.append({"ref_idx": fid,
                             "gt_file": f"{fid}_gt.npy",
                             "gen_file": f"{fid}_gen.npy"})
            idx += 1

    (outdir / "generation_manifest.json").write_text(
        json.dumps({"total": len(manifest), "samples": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[frequency_gen] done: {len(manifest)} samples -> {outdir}")


if __name__ == "__main__":
    main()
