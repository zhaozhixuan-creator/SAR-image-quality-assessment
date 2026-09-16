"""Stage B —— 生成：调度各模型推理，产出待评估的生成图像 + generation_manifest.json。

- stylegan4sar：两变体（基线/增强）按 (类, 角度) 条件批量生成 128×128。
- angle_gen：两 checkpoint（geometry/baseline）NVS 生成 64×64 目标视角 + 参考/真值。
- frequency_gen：X→Ka Pix2Pix 实跑推理（真实 wholeImg 测试集 99 对，生成真实 Ka vs 生成 Ka）。
- gaussrecon4sar：本机无法重生成，导入预生成 renders(生成)/gt(真值) 成对结果。
"""
from __future__ import annotations

import numpy as np
import subprocess
from pathlib import Path

from common import io, paths


def _venv_py(cfg, model_key: str) -> str:
    return paths.resolve(cfg["models"][model_key]["venv_python"], cfg)


def _gen_dir(cfg, *parts) -> Path:
    return paths.ws_dir(cfg, "generated", *parts)


def run_stylegan(cfg, args) -> None:
    m = cfg["models"]["stylegan4sar"]
    angles_json = paths.ws_dir(cfg, "real", str(m["size"]), "train", "labels.json")
    for variant, v in m["variants"].items():
        out = _gen_dir(cfg, "stylegan4sar", variant)
        cmd = [
            _venv_py(cfg, "stylegan4sar"),
            str(paths.V3_ROOT / "generators" / "stylegan_generate.py"),
            "--checkpoint", paths.resolve(v["checkpoint"], cfg),
            "--outdir", str(out),
            "--angles-json", str(angles_json),
            "--per-class", str(m["per_class"]),
            "--aasg-enabled", "1" if v["aasg_enabled"] else "0",
            "--classes", str(len(cfg["datasets"]["class_names"])),
            "--device", f"cuda:{cfg['evaluation']['gpu']}",
        ]
        print(f"[stage_b] stylegan4sar/{variant}: {v['note']}")
        subprocess.run(cmd, cwd=paths.resolve(m["cwd"], cfg), check=True)
        # 回填类名
        man = io.read_json(out / "generation_manifest.json")
        for s in man["samples"]:
            s["class_name"] = cfg["datasets"]["class_names"][s["class_idx"]]
        io.write_json(out / "generation_manifest.json", man)


def _write_angle_gen_config(cfg, checkpoint: str, network_base: str, gpu: int, out_yml: Path) -> None:
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


def run_angle_gen(cfg, args) -> None:
    m = cfg["models"]["angle_gen"]
    cfg_dir = _gen_dir(cfg, "angle_gen", "_configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for variant, v in m["variants"].items():
        out = _gen_dir(cfg, "angle_gen", variant)
        yml = cfg_dir / f"{variant}.yml"
        _write_angle_gen_config(cfg, paths.resolve(v["checkpoint"], cfg),
                                v.get("network_base", "xunet"),
                                cfg["evaluation"]["gpu"], yml)
        cmd = [
            _venv_py(cfg, "angle_gen"),
            str(paths.V3_ROOT / "generators" / "angle_gen_generate.py"),
            "--config", str(yml),
            "--outdir", str(out),
            "--per-class", str(m["per_class"]),
            "--test-target-type", "all",
        ]
        print(f"[stage_b] angle_gen/{variant}: {v['note']}")
        subprocess.run(cmd, cwd=paths.resolve(m["cwd"], cfg), check=True)


def run_frequency_gen(cfg, args) -> None:
    m = cfg["models"]["frequency_gen"]
    out = _gen_dir(cfg, "frequency_gen")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        _venv_py(cfg, "frequency_gen"),
        str(paths.V3_ROOT / "generators" / "frequency_gen_generate.py"),
        "--config", paths.resolve(m["config"], cfg),
        "--checkpoint", paths.resolve(m["checkpoint"], cfg),
        "--outdir", str(out),
        "--gpu", str(cfg["evaluation"]["gpu"]),
    ]
    if m.get("data_root"):
        cmd += ["--dataroot", paths.resolve(m["data_root"], cfg)]
    print(f"[stage_b] frequency_gen: {m['note']}")
    subprocess.run(cmd, cwd=paths.resolve(m["cwd"], cfg), check=True)
    # 回填元数据（真实数据 → 不再标记 data_limited，报告按适用指标正常出表）
    man = io.read_json(out / "generation_manifest.json")
    man["note"] = m["note"]
    man["data_limited"] = False
    io.write_json(out / "generation_manifest.json", man)


def run_gaussrecon(cfg, args) -> None:
    m = cfg["models"]["gaussrecon4sar"]
    base = Path(paths.resolve(m["source_root"], cfg))
    for variant, v in m["variants"].items():
        out = _gen_dir(cfg, "gaussrecon4sar", variant)
        out.mkdir(parents=True, exist_ok=True)
        renders_dir = base / v["exp"] / "train" / v["iteration"] / "renders"
        gt_dir = base / v["exp"] / "train" / v["iteration"] / "gt"
        samples = []
        for rp in sorted(renders_dir.glob("*.png")):
            gp = gt_dir / rp.name
            if not gp.exists():
                continue
            # 文件名约定：{俯仰角}_{方位角}_{类别}.png，如 17_182.790649_T72.png
            parts = rp.stem.split("_")
            depression = float(parts[0])
            azimuth = float(parts[1])
            cls = parts[2] if len(parts) > 2 else "?"
            ridx = len(samples)
            np.save(out / f"{ridx:04d}_gt.npy", io.read_gray_01(gp).astype(np.float32))
            np.save(out / f"{ridx:04d}_gen.npy", io.read_gray_01(rp).astype(np.float32))
            samples.append({
                "ref_idx": f"{ridx:04d}",
                "azimuth_deg": azimuth,
                "depression_deg": depression,
                "class_name": cls,
                "gt_file": f"{ridx:04d}_gt.npy",
                "gen_file": f"{ridx:04d}_gen.npy",
            })
        io.write_json(out / "generation_manifest.json",
                      {"total": len(samples), "samples": samples, "note": v.get("note", "")})
        print(f"[stage_b] gaussrecon4sar/{variant}: 导入 {len(samples)} 对 renders/gt")


def run(cfg, args) -> None:
    only = args.model if args and getattr(args, "model", None) else None
    def do(key, fn):
        if only and only != key:
            return
        fn(cfg, args)

    do("stylegan4sar", run_stylegan)
    do("angle_gen", run_angle_gen)
    do("frequency_gen", run_frequency_gen)
    do("gaussrecon4sar", run_gaussrecon)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    run(paths.load_config(), None)
