#!/usr/bin/env python3
"""三个「质检有效性」验证实验（专利实证）。

  --exp 1  重选模型（免费，复用 v3 既有指标）：angle_gen 三变体 → 改派 baseline，验证 ΔENL 改善
  --exp 2  调参（GPU ~30min）：angle_gen 输出全去噪 w 栈，逐步质检「扩散步数→质量」单调性
  --exp 3  重训（GPU ~1-2h）：stylegan 快照续训 → 新 checkpoint 复评，验证 FID/SSIM/AFS 改善

所有证据（指标 JSON + 改进幅度 + 日志）写入 closed_loop/results/experiments/，
供后续按专利模板回填。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.io_utils import ensure_dir, load_config, write_json
from evaluation.metrics_bridge import load_existing_metrics, v3_root, load_v3_config, compute_metrics
from evaluation.portrait import ALL_KEYS
from feedback.reselect import reselect
from feedback.adjust_param import analyze_sweep

RESULT_DIR = ensure_dir(PROJECT_ROOT / "results" / "experiments")


# ---------------------------------------------------------------------------
# 实验 1：重选模型（免费）
# ---------------------------------------------------------------------------
def exp1_reselect() -> dict:
    existing = load_existing_metrics()
    angle_variants = {v: existing[f"angle_gen/{v}"] for v in ["geometry", "baseline", "adapted"]
                      if f"angle_gen/{v}" in existing}

    req = {"background_consistency": "high", "structure_fidelity": "medium", "azimuth_precision": "low"}
    result = reselect("angle_gen", "adapted", req, angle_variants)

    # 结构保真维度也测一下（第二个关注维度）
    req2 = {"background_consistency": "medium", "structure_fidelity": "high", "azimuth_precision": "low"}
    result2 = reselect("angle_gen", "adapted", req2, angle_variants)

    payload = {
        "experiment": "exp1_reselect",
        "mechanism": "模型/变体重选",
        "question": "质检能否区分变体并指导选型，反馈后指标是否改善？",
        "available_variants": {v: {k: m.get(k) for k in ["delta_enl", "bve", "ssim", "afs", "psnr"]}
                               for v, m in angle_variants.items()},
        "case_background_consistency": result,
        "case_structure_fidelity": result2,
        "conclusion": "reselect 有效" if result.get("applied") else "reselect 未触发",
    }
    write_json(RESULT_DIR / "exp1_reselect.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# ---------------------------------------------------------------------------
# 实验 2：调参（angle_gen 扩散步数）
# ---------------------------------------------------------------------------
def _angle_gen_yml(checkpoint: str, network_base: str, gpu: int, out_yml: Path) -> None:
    txt = f"""name: diffusion_ori_v3
use_tb_logger: false

datasets:
  base: mstar-soc10

gpu_ids: [{gpu}]
scheme: supervised
sub_scheme: original
pretrain: pretrained
setting:
  base: test

network:
  base: {network_base}

pipeline: diffusion
sub_pipeline: original
pipL:
  logsnr_min: -20.0
  logsnr_max: 20.0
  total_timestep: 256
  w: [0, 1, 2, 3, 4, 5, 6, 7]

path:
  pretrain_model_pth: {checkpoint}
"""
    out_yml.write_text(txt, encoding="utf-8")


def _pair_ssim_psnr(gt, gen):
    """单对 (H,W) float32 [0,1] → ssim/psnr/mse/ncc。"""
    import numpy as np
    from skimage.metrics import structural_similarity

    a = np.asarray(gt, dtype=np.float64)
    b = np.asarray(gen, dtype=np.float64)
    gmax = max(float(a.max()), float(b.max()), 1e-9)
    a, b = a / gmax, b / gmax
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(structural_similarity(a, b, data_range=1.0))
    aa, bb = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = float(np.sqrt((aa @ aa) * (bb @ bb)))
    ncc = 0.0 if den <= 0 else float((aa @ bb) / den)
    return {"ssim": ssim, "psnr": psnr, "mse": mse, "ncc": ncc}


def exp2_adjust_param(gpu: int = 2, per_class: int = 3) -> dict:
    import numpy as np

    cfg = load_v3_config()
    m = cfg["models"]["angle_gen"]
    venv_py = m["venv_python"].replace("{root}", cfg["root"])
    cwd = m["cwd"].replace("{root}", cfg["root"])
    gen_dir = ensure_dir(PROJECT_ROOT / "results" / "generated" / "angle_gen_wstack")
    cfg_dir = ensure_dir(PROJECT_ROOT / "results" / "_cfg")
    yml = cfg_dir / "geometry_savew.yml"
    v = m["variants"]["geometry"]
    _angle_gen_yml(v["checkpoint"].replace("{root}", cfg["root"]),
                   v.get("network_base", "xunet"), gpu, yml)

    print("[exp2] 生成 angle_gen 全 w 栈（--save-all-w）…", flush=True)
    subprocess.run([
        venv_py, str(v3_root() / "generators" / "angle_gen_generate.py"),
        "--config", str(yml), "--outdir", str(gen_dir),
        "--per-class", str(per_class), "--test-target-type", "all", "--save-all-w",
    ], cwd=cwd, check=True)

    man = json.loads((gen_dir / "generation_manifest.json").read_text(encoding="utf-8"))
    n_w = len(man["samples"][0]["w_files"]) if man["samples"] else 0
    per_w = {k: {"ssim": [], "psnr": [], "mse": [], "ncc": []} for k in range(n_w)}
    for s in man["samples"]:
        gt = np.load(gen_dir / s["gt_file"])
        for k, wf in enumerate(s["w_files"]):
            gen = np.load(gen_dir / wf)
            r = _pair_ssim_psnr(gt, gen)
            for key in per_w[k]:
                per_w[k][key].append(r[key])

    agg = {k: {key: float(np.mean(v)) for key, v in per_w[k].items()} for k in per_w}
    sweep = {"ssim": analyze_sweep(list(agg), agg, "ssim"),
             "psnr": analyze_sweep(list(agg), agg, "psnr"),
             "ncc": analyze_sweep(list(agg), agg, "ncc")}

    payload = {
        "experiment": "exp2_adjust_param",
        "mechanism": "生成参数调整（扩散去噪步数）",
        "question": "质检指标是否随扩散步数单调变化（即质检对生成参数敏感、可指导调参）？",
        "n_samples": len(man["samples"]), "n_w": n_w,
        "per_step_metrics": agg,
        "analysis": sweep,
        "conclusion": "质检对扩散步数敏感" if sweep["ssim"]["monotonic"] else "未观察到单调性",
    }
    write_json(RESULT_DIR / "exp2_adjust_param.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# ---------------------------------------------------------------------------
# 实验 3：重训（stylegan 续训）
# ---------------------------------------------------------------------------
def _stylegan_env(cfg, gpu: int) -> dict:
    model_root = cfg["models"]["stylegan4sar"]["cwd"].replace("{root}", cfg["root"])
    env = os.environ.copy()
    env.update({
        "PATH": f"{model_root}/.venv/bin:{env.get('PATH', '')}",
        "CUDA_HOME": f"{model_root}/.venv",
        "LD_LIBRARY_PATH": f"{model_root}/.venv/lib:{env.get('LD_LIBRARY_PATH', '')}",
        "CC": "/usr/bin/gcc-11", "CXX": "/usr/bin/g++-11",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        # 原 /tmp/sar_stylegan4sar_torch_extensions 属另一用户 zqm 且不可写，导致 CUDA kernel
        # 回退到慢速参考实现；改为当前用户可写的目录以编译原生 kernel。
        "TORCH_EXTENSIONS_DIR": "/tmp/sar_stylegan4sar_torch_ext_zzx",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    })
    return model_root, env


def exp3_retrain(gpu: int = 2, kimg: int = 200) -> dict:
    cfg = load_v3_config()
    m = cfg["models"]["stylegan4sar"]
    model_root, env = _stylegan_env(cfg, gpu)
    venv_py = m["venv_python"].replace("{root}", cfg["root"])
    variant = m["variants"]["stylegan2_ada_5class"]
    resume_pkl = variant["checkpoint"].replace("{root}", cfg["root"])
    data_zip = f"{model_root}/datasets/mstar15_5class_stride5_maxdev-128.zip"
    outdir = str(ensure_dir(PROJECT_ROOT / "results" / "retrain" / "stylegan"))

    print(f"[exp3] 续训 stylegan --resume={resume_pkl} --kimg={kimg} …", flush=True)
    subprocess.run([
        venv_py, "train.py", f"--outdir={outdir}", f"--data={data_zip}",
        "--gpus=1", "--cond=1", "--cfg=auto", "--aug=ada", "--batch=32",
        f"--kimg={kimg}", "--snap=25", "--seed=0", "--metrics=none", "--workers=3",
        f"--resume={resume_pkl}",
    ], cwd=model_root, env=env, check=True)

    # 找最新 snapshot
    snapshots = sorted(Path(outdir).rglob("network-snapshot-*.pkl"),
                       key=lambda p: int(p.stem.split("-")[-1]))
    if not snapshots:
        raise RuntimeError("续训未产出 network-snapshot-*.pkl")
    new_pkl = snapshots[-1]

    print(f"[exp3] 用新 snapshot 生成：{new_pkl} …", flush=True)
    gen_dir = ensure_dir(PROJECT_ROOT / "results" / "generated" / "stylegan_finetuned")
    # 复用 v3 的 stylegan 生成封装（复用真实角度 labels_stylegan.json 需先 prepare）
    angles_json = v3_root() / "workspace" / "real" / "128" / "train" / "labels_stylegan.json"
    if not angles_json.exists():
        raise RuntimeError("缺少 labels_stylegan.json，请先运行 v3 --stage prepare/generate")
    classes = m.get("classes") or cfg["datasets"]["class_names"]
    subprocess.run([
        venv_py, str(v3_root() / "generators" / "stylegan_generate.py"),
        "--checkpoint", str(new_pkl), "--outdir", str(gen_dir),
        "--angles-json", str(angles_json), "--per-class", str(m["per_class"]),
        "--aasg-enabled", "0", "--classes", str(len(classes)),
        "--device", f"cuda:{gpu}",
    ], cwd=m["cwd"].replace("{root}", cfg["root"]), check=True)

    # 回填类名 + 评估（委托 v3 的 align + evaluate 逻辑，见下方 _evaluate_stylegan）
    man_path = gen_dir / "generation_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    for s in man["samples"]:
        s["class_name"] = classes[s["class_idx"]]
    write_json(man_path, man)
    new_metrics = _evaluate_stylegan(cfg, gen_dir, classes, gpu)

    existing = load_existing_metrics()
    baseline = existing.get("stylegan4sar/stylegan2_ada_5class", {})

    payload = {
        "experiment": "exp3_retrain",
        "mechanism": "重训（stylegan2-ada resume 续训）",
        "question": "质检发现的弱点（结构保真弱）经重训后，指标是否改善？",
        "resume_checkpoint": resume_pkl,
        "new_checkpoint": str(new_pkl),
        "kimg": kimg,
        "baseline_metrics": {k: baseline.get(k) for k in ALL_KEYS},
        "retrained_metrics": new_metrics,
        "delta": {k: (new_metrics.get(k) - baseline.get(k) if (new_metrics.get(k) is not None and baseline.get(k) is not None) else None)
                  for k in ALL_KEYS},
        "conclusion": "见 delta 列",
    }
    write_json(RESULT_DIR / "exp3_retrain.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _evaluate_stylegan(cfg, gen_dir: Path, classes: list, gpu: int) -> dict:
    """对齐 + 13 指标（复用 v3 的 metrics_bridge + R/E_asc）。需在 v3 评估环境运行。"""
    import numpy as np
    sys.path.insert(0, str(v3_root()))
    # R/E_asc.pt 的 unpickle 需要 v2 gan_metrics 可导入
    v2_root = v3_root().parent / "SAR-image-quality-assessment-v2"
    if str(v2_root) not in sys.path:
        sys.path.insert(0, str(v2_root))
    from common import io as v3_io
    from common import paths as v3_paths

    size = 128
    real_img = np.load(v3_paths.ws_dir(cfg, "real", str(size), "train", "images.npy"))
    real_lab = v3_io.read_json(v3_paths.ws_dir(cfg, "real", str(size), "train", "labels.json"))
    angles_by_class = {c: [] for c in classes}
    for i, lb in enumerate(real_lab):
        if lb["class_name"] in angles_by_class:
            angles_by_class[lb["class_name"]].append((lb["azimuth_deg"], i))

    man = v3_io.read_json(gen_dir / "generation_manifest.json")
    real_l, fake_l, ang_l = [], [], []
    for s in man["samples"]:
        cn = s["class_name"]
        ri = min(angles_by_class[cn], key=lambda x: abs(x[0] - s["azimuth_deg"]))[1]
        real_l.append(real_img[ri])
        fake_l.append(v3_io.read_gray_01(gen_dir / s["filename"]))
        ang_l.append(s["azimuth_deg"])

    ckpt = v3_paths.ws_dir(cfg, "checkpoints")
    import torch
    R = torch.load(ckpt / "R_128.pt", map_location="cpu")
    E = torch.load(ckpt / "E_asc.pt", map_location="cpu")
    device = f"cuda:{gpu}"
    R, E = R.to(device), E.to(device)
    out = compute_metrics(real_l, fake_l, ang_l, R, E, size=size, device=device)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    cli = argparse.ArgumentParser(description="质检有效性验证实验。")
    cli.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    cli.add_argument("--gpu", type=int, default=2)
    cli.add_argument("--per-class", type=int, default=3, help="exp2 每类样本数")
    cli.add_argument("--kimg", type=int, default=200, help="exp3 续训 kimg")
    args = cli.parse_args()

    if args.exp == 1:
        exp1_reselect()
    elif args.exp == 2:
        exp2_adjust_param(gpu=args.gpu, per_class=args.per_class)
    elif args.exp == 3:
        exp3_retrain(gpu=args.gpu, kimg=args.kimg)


if __name__ == "__main__":
    main()
