"""指标桥接：复用 v2 `gan_metrics`，在统一评估环境下计算 6 项论文指标 + 7 项 FR 指标。

不复制 v2 实现，仅通过 sys.path 指向 v2 的 gan_metrics 包导入。对调用方暴露：
- load_gm(cfg)          -> gan_metrics 模块
- compute_all(gm, real, fake, angles_deg, R, E_asc, ...) -> dict（六项 + 七项）

数据契约：real / fake 均为 list[np.ndarray] 幅度域 [0,1] 灰度图，逐对对齐（长度一致）。
ΔENL / BVE 在强度域（幅度²）计算（与 v2 论文口径一致）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import paths

PAPER_KEYS = ["fid", "ssim", "afs", "delta_enl", "bve", "cmae_deg"]
FR_KEYS = ["mse", "rmse", "psnr", "ncc", "uqi", "ms_ssim", "fsim"]
ALL_KEYS = PAPER_KEYS + FR_KEYS


def load_gm(cfg):
    """导入 v2 gan_metrics 并返回模块。"""
    gm_dir = Path(paths.resolve(cfg["v2_gan_metrics"], cfg))      # .../gan_metrics
    v2_root = str(gm_dir.parent)                                   # .../SAR-image-quality-assessment-v2
    if v2_root not in sys.path:
        sys.path.insert(0, v2_root)
    import gan_metrics as gm
    return gm


def compute_paper(gm, real, fake, angles_deg, R, E_asc,
                  size: int = 128, device: str = "cpu", fid_batch: int = 16,
                  pca_dim: int = 64, metrics=None) -> dict:
    """六项论文指标（FID/SSIM/AFS/ΔENL/BVE/CMAE）。

    `metrics`：需要的指标子集（None=全部六项）。不适用/不可算的项返回 None：
      - FID 需 ≥2 样本（否则协方差退化）
      - CMAE 需有角度标注（angles_deg 非空）
    仍固定返回全部 6 个键，未计算的键置 None，保证报告 schema 稳定。
    """
    want = set(PAPER_KEYS) if metrics is None else set(metrics) & set(PAPER_KEYS)
    out = {}
    # 强度域（幅度²）用于背景统计
    real_i = [r ** 2 for r in real]
    fake_i = [f ** 2 for f in fake]

    out["fid"] = (float(gm.fid(real, fake, batch=fid_batch, pca_dim=pca_dim, device=device))
                  if ("fid" in want and len(real) >= 2 and len(fake) >= 2) else None)
    out["ssim"] = float(gm.ssim(real, fake)) if "ssim" in want else None
    out["afs"] = float(gm.afs(real, fake, E_asc, device=device)) if "afs" in want else None
    out["delta_enl"] = float(gm.delta_enl(real_i, fake_i)) if "delta_enl" in want else None
    out["bve"] = float(gm.bve(real_i, fake_i)) if "bve" in want else None

    if "cmae_deg" in want and angles_deg is not None and len(angles_deg) > 0:
        est_angles = gm.estimate_angles(fake, R, size=size, device=device)
        out["cmae_deg"] = float(gm.cmae(est_angles, angles_deg))
    else:
        out["cmae_deg"] = None
    return out


def compute_fr(gm, real, fake, metrics=None) -> dict:
    """七项全参考指标（MSE/RMSE/PSNR/NCC/UQI/MS-SSIM/FSIM）。不适用项置 None。"""
    want = set(FR_KEYS) if metrics is None else set(metrics) & set(FR_KEYS)
    fns = {"mse": gm.mse, "rmse": gm.rmse, "psnr": gm.psnr, "ncc": gm.ncc,
           "uqi": gm.uqi, "ms_ssim": gm.ms_ssim, "fsim": gm.fsim}
    return {k: (float(fns[k](real, fake)) if k in want else None) for k in FR_KEYS}


def compute_all(gm, real, fake, angles_deg, R, E_asc, size: int = 128,
                device: str = "cpu", fid_batch: int = 16, pca_dim: int = 64,
                metrics=None) -> dict:
    paper = compute_paper(gm, real, fake, angles_deg, R, E_asc,
                          size=size, device=device, fid_batch=fid_batch, pca_dim=pca_dim,
                          metrics=metrics)
    fr = compute_fr(gm, real, fake, metrics=metrics)
    return {**paper, **fr}
