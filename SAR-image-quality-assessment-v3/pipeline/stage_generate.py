"""Stage B —— 生成：调度各模型推理，产出待评估的生成图像 + generation_manifest.json。

- stylegan4sar：StyleGAN2-ADA 基线按 (类, 角度) 条件批量生成 128×128（5 类）。
- angle_gen：两 checkpoint（geometry/baseline）NVS 生成 64×64 目标视角 + 参考/真值。
- frequency_gen：X→Ka Pix2Pix 实跑推理（真实 wholeImg 测试集 99 对，生成真实 Ka vs 生成 Ka）。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from common import io, paths


def _venv_py(cfg, model_key: str) -> str:
    return paths.resolve(cfg["models"][model_key]["venv_python"], cfg)


def _gen_dir(cfg, *parts) -> Path:
    return paths.ws_dir(cfg, "generated", *parts)


def _stylegan_angles_json(cfg, m, classes):
    """从全局 10 类真实 labels.json 抽取 StyleGAN 需要的 5 类角度，并按类顺序重排下标。"""
    src = paths.ws_dir(cfg, "real", str(m["size"]), "train", "labels.json")
    labs = io.read_json(src)
    out = []
    for ci, cn in enumerate(classes):
        for lb in labs:
            if lb["class_name"] == cn:
                out.append({"class_idx": ci, "class_name": cn,
                            "azimuth_deg": lb["azimuth_deg"]})
    dst = paths.ws_dir(cfg, "real", str(m["size"]), "train", "labels_stylegan.json")
    io.write_json(dst, out)
    return dst


def run_stylegan(cfg, args) -> None:
    m = cfg["models"]["stylegan4sar"]
    classes = m.get("classes") or cfg["datasets"]["class_names"]
    angles_json = _stylegan_angles_json(cfg, m, classes)
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
            "--classes", str(len(classes)),
            "--device", f"cuda:{cfg['evaluation']['gpu']}",
        ]
        print(f"[stage_b] stylegan4sar/{variant}: {v['note']}")
        subprocess.run(cmd, cwd=paths.resolve(m["cwd"], cfg), check=True)
        # 回填类名
        man = io.read_json(out / "generation_manifest.json")
        for s in man["samples"]:
            s["class_name"] = classes[s["class_idx"]]
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


def run(cfg, args) -> None:
    only = args.model if args and getattr(args, "model", None) else None
    def do(key, fn):
        if only and only != key:
            return
        fn(cfg, args)

    do("stylegan4sar", run_stylegan)
    do("angle_gen", run_angle_gen)
    do("frequency_gen", run_frequency_gen)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    run(paths.load_config(), None)
