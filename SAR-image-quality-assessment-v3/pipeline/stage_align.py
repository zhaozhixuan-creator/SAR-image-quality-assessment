"""Stage C —— 对齐：为每个模型/变体构建「真实↔生成」逐对对齐集，并训练辅助网络 R/E_asc。

输出（workspace/aligned/{model}/{variant}/）：
  - real.npy    (N,H,W) float32 [0,1] —— 与生成图配对/同分布的真实图
  - fake.npy    (N,H,W) float32 [0,1] —— 生成图
  - angles.npy  (N,)   float —— 生成图的真实/条件角度（CMAE 用）

辅助网络（workspace/checkpoints/）：R_128.pt / R_64.pt / E_asc.pt（在真实训练集上训练）。
"""
from __future__ import annotations

import numpy as np
import torch

from common import io, paths


def _train_helpers(gm, cfg):
    ckpt = paths.ws_dir(cfg, "checkpoints")
    ckpt.mkdir(parents=True, exist_ok=True)
    device = paths.resolve_device(cfg)

    needed = [f"R_{size}.pt" for size in cfg["datasets"]["sizes"]] + ["E_asc.pt"]
    if all((ckpt / n).exists() for n in needed):
        print(f"[stage_c] 辅助网络已存在（{', '.join(needed)}），跳过训练（删掉对应 .pt 可强制重训）")
        return

    ev = cfg["evaluation"]
    for size in cfg["datasets"]["sizes"]:
        img = np.load(paths.ws_dir(cfg, "real", str(size), "train", "images.npy"))
        labels = io.read_json(paths.ws_dir(cfg, "real", str(size), "train", "labels.json"))
        ang = np.asarray([lb["azimuth_deg"] for lb in labels], dtype=np.float64)
        print(f"[stage_c] 训练 R(size={size}) on {len(img)} 张 …")
        R = gm.train_angle_estimator(list(img), ang.tolist(),
                                     epochs=ev["angle_estimator_epochs"],
                                     size=size, device=device, seed=ev["seed"])
        torch.save(R, ckpt / f"R_{size}.pt")

    # E_asc 共享（size=64），在 128 真实集上取目标区训练
    img128 = np.load(paths.ws_dir(cfg, "real", "128", "train", "images.npy"))
    print(f"[stage_c] 训练 E_asc on {len(img128)} 张 …")
    E = gm.train_asc_extractor(list(img128), epochs=ev["asc_epochs"],
                               size=64, device=device, seed=ev["seed"])
    torch.save(E, ckpt / "E_asc.pt")
    print(f"[stage_c] 辅助网络已保存到 {ckpt}")


def _nearest_idx(angles_by_class, ci, target):
    arr = angles_by_class[ci]  # list[(angle, img_idx)]
    best = min(arr, key=lambda x: abs(x[0] - target))
    return best[1]


def _align_stylegan(cfg):
    m = cfg["models"]["stylegan4sar"]
    size = m["size"]
    real_img = np.load(paths.ws_dir(cfg, "real", str(size), "train", "images.npy"))
    real_lab = io.read_json(paths.ws_dir(cfg, "real", str(size), "train", "labels.json"))

    angles_by_class = {ci: [] for ci in range(len(cfg["datasets"]["class_names"]))}
    for i, lb in enumerate(real_lab):
        angles_by_class[lb["class_idx"]].append((lb["azimuth_deg"], i))

    for variant in m["variants"]:
        gen_dir = paths.ws_dir(cfg, "generated", "stylegan4sar", variant)
        man = io.read_json(gen_dir / "generation_manifest.json")
        real_l, fake_l, ang_l = [], [], []
        for s in man["samples"]:
            ci = s["class_idx"]
            ri = _nearest_idx(angles_by_class, ci, s["azimuth_deg"])
            fake = io.read_gray_01(gen_dir / s["filename"])
            real_l.append(real_img[ri])
            fake_l.append(fake)
            ang_l.append(s["azimuth_deg"])
        out = paths.ws_dir(cfg, "aligned", "stylegan4sar", variant)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "real.npy", np.stack(real_l).astype(np.float32))
        np.save(out / "fake.npy", np.stack(fake_l).astype(np.float32))
        np.save(out / "angles.npy", np.asarray(ang_l, dtype=np.float64))
        print(f"[stage_c] stylegan4sar/{variant}: 对齐 {len(real_l)} 对")


def _align_angle_gen(cfg):
    m = cfg["models"]["angle_gen"]
    for variant in m["variants"]:
        gen_dir = paths.ws_dir(cfg, "generated", "angle_gen", variant)
        man_path = gen_dir / "generation_manifest.json"
        if not man_path.exists():
            print(f"[stage_c] angle_gen/{variant}: 无生成结果，跳过")
            continue
        man = io.read_json(man_path)
        real_l, fake_l, ang_l = [], [], []
        for s in man["samples"]:
            gt = np.load(gen_dir / s["gt_file"])
            gen = np.load(gen_dir / s["gen_file"])
            real_l.append(gt)
            fake_l.append(gen)
            ang_l.append(s["tgt_azimuth_deg"])
        out = paths.ws_dir(cfg, "aligned", "angle_gen", variant)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "real.npy", np.stack(real_l).astype(np.float32))
        np.save(out / "fake.npy", np.stack(fake_l).astype(np.float32))
        np.save(out / "angles.npy", np.asarray(ang_l, dtype=np.float64))
        print(f"[stage_c] angle_gen/{variant}: 对齐 {len(real_l)} 对")


def _align_frequency_gen(cfg):
    """X→Ka 翻译：real = 真值 Ka(gt)，fake = 生成 Ka(gen)；无角度条件 → 不写 angles.npy。"""
    gen_dir = paths.ws_dir(cfg, "generated", "frequency_gen")
    man_path = gen_dir / "generation_manifest.json"
    if not man_path.exists():
        print("[stage_c] frequency_gen: 无生成结果，跳过")
        return
    man = io.read_json(man_path)
    real_l, fake_l = [], []
    for s in man["samples"]:
        real_l.append(np.load(gen_dir / s["gt_file"]))
        fake_l.append(np.load(gen_dir / s["gen_file"]))
    out = paths.ws_dir(cfg, "aligned", "frequency_gen", "wholeimg")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "real.npy", np.stack(real_l).astype(np.float32))
    np.save(out / "fake.npy", np.stack(fake_l).astype(np.float32))
    print(f"[stage_c] frequency_gen/wholeimg: 对齐 {len(real_l)} 对（无角度，仅 FR/SSIM）")


def _align_gaussrecon(cfg):
    """3DGS 重构：real = 真值视图(gt)，fake = 渲染视图(render)；角度取自文件名。"""
    m = cfg["models"]["gaussrecon4sar"]
    for variant in m["variants"]:
        gen_dir = paths.ws_dir(cfg, "generated", "gaussrecon4sar", variant)
        man_path = gen_dir / "generation_manifest.json"
        if not man_path.exists():
            print(f"[stage_c] gaussrecon4sar/{variant}: 无生成结果，跳过")
            continue
        man = io.read_json(man_path)
        real_l, fake_l, ang_l = [], [], []
        for s in man["samples"]:
            real_l.append(np.load(gen_dir / s["gt_file"]))
            fake_l.append(np.load(gen_dir / s["gen_file"]))
            ang_l.append(s["azimuth_deg"])
        out = paths.ws_dir(cfg, "aligned", "gaussrecon4sar", variant)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "real.npy", np.stack(real_l).astype(np.float32))
        np.save(out / "fake.npy", np.stack(fake_l).astype(np.float32))
        np.save(out / "angles.npy", np.asarray(ang_l, dtype=np.float64))
        print(f"[stage_c] gaussrecon4sar/{variant}: 对齐 {len(real_l)} 对")


def run(cfg, args):
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    from evaluation.metrics_bridge import load_gm
    gm = load_gm(cfg)

    _train_helpers(gm, cfg)
    _align_stylegan(cfg)
    _align_angle_gen(cfg)
    _align_frequency_gen(cfg)
    _align_gaussrecon(cfg)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    run(paths.load_config(), None)
