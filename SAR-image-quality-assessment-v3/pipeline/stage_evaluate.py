"""Stage D —— 评估：对每个对齐集调用 v2 指标（6 项论文 + 7 项 FR），写 results/metrics/*/*.json。

指标适用矩阵（按模型 `metrics` 配置裁剪；不适用项置 None）：
  - stylegan4sar / angle_gen / gaussrecon4sar：全部 13 项
  - frequency_gen（X→Ka 翻译）：仅 SSIM + 7 项 FR（无角度/分布域不匹配）
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common import io, paths


def _load_helpers(cfg):
    ckpt = paths.ws_dir(cfg, "checkpoints")
    R_128 = torch.load(ckpt / "R_128.pt", map_location="cpu")
    R_64 = torch.load(ckpt / "R_64.pt", map_location="cpu")
    E_asc = torch.load(ckpt / "E_asc.pt", map_location="cpu")
    return R_128, R_64, E_asc


def _fmt(v, prec=4):
    return "—" if v is None else f"{v:.{prec}f}"


def _evaluate_one(gm, cfg, model, variant, size, R, E_asc, device, meta: dict, metrics=None):
    d = paths.ws_dir(cfg, "aligned", model, variant)
    real = [x for x in np.load(d / "real.npy")]
    fake = [x for x in np.load(d / "fake.npy")]
    ang_path = d / "angles.npy"
    ang = np.load(ang_path).tolist() if ang_path.exists() else None

    from evaluation.metrics_bridge import compute_all
    out = compute_all(gm, real, fake, ang, R, E_asc, size=size, device=device,
                      fid_batch=cfg["evaluation"]["fid_batch"],
                      pca_dim=cfg["evaluation"]["fid_pca_dim"],
                      metrics=metrics)
    out.update({"model": model, "variant": variant, "size": size,
                "n_pairs": len(real), "note": meta.get("note", "")})
    res = paths.res_dir(cfg, "metrics", model)
    res.mkdir(parents=True, exist_ok=True)
    io.write_json(res / f"{variant}.json", out)
    print(f"[stage_d] {model}/{variant}: n={len(real)} "
          f"FID={_fmt(out.get('fid'), 2)} SSIM={_fmt(out.get('ssim'))} "
          f"AFS={_fmt(out.get('afs'))} ΔENL={_fmt(out.get('delta_enl'))} "
          f"BVE={_fmt(out.get('bve'))} CMAE={_fmt(out.get('cmae_deg'), 2)}°")


def _metrics_for(cfg, model_key):
    m = cfg["models"][model_key]
    return m.get("metrics")  # None → 全部 13 项


def run(cfg, args):
    sys.path.insert(0, str(paths.V3_ROOT))
    from evaluation.metrics_bridge import load_gm
    gm = load_gm(cfg)
    device = paths.resolve_device(cfg)
    R_128, R_64, E_asc = _load_helpers(cfg)
    R_128 = R_128.to(device)
    R_64 = R_64.to(device)
    E_asc = E_asc.to(device)

    only = args.model if args and getattr(args, "model", None) else None

    if not only or only == "stylegan4sar":
        for variant, v in cfg["models"]["stylegan4sar"]["variants"].items():
            _evaluate_one(gm, cfg, "stylegan4sar", variant, 128, R_128, E_asc, device, v,
                          _metrics_for(cfg, "stylegan4sar"))

    if not only or only == "angle_gen":
        for variant, v in cfg["models"]["angle_gen"]["variants"].items():
            _evaluate_one(gm, cfg, "angle_gen", variant, 64, R_64, E_asc, device, v,
                          _metrics_for(cfg, "angle_gen"))

    if not only or only == "frequency_gen":
        _evaluate_one(gm, cfg, "frequency_gen", "smoke", 64, R_64, E_asc, device,
                      cfg["models"]["frequency_gen"], _metrics_for(cfg, "frequency_gen"))

    if not only or only == "gaussrecon4sar":
        for variant, v in cfg["models"]["gaussrecon4sar"]["variants"].items():
            _evaluate_one(gm, cfg, "gaussrecon4sar", variant, 128, R_128, E_asc, device, v,
                          _metrics_for(cfg, "gaussrecon4sar"))


if __name__ == "__main__":
    sys.path.insert(0, str(paths.V3_ROOT))
    run(paths.load_config(), None)
