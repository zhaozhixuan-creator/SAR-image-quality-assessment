import argparse
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dnnlib
import legacy
import numpy as np
import torch
from PIL import Image


DATASET_ZIP = PROJECT_ROOT / "datasets" / "t32_t72-128.zip"
TRAINING_RUNS_DIR = PROJECT_ROOT / "training-runs"
DEFAULT_NETWORK_NAME = "network-snapshot-000800.pkl"
CLASS_SPECS: Tuple[Tuple[str, int], ...] = (("T32", 0), ("T72", 1))
ANGLE_START_DEG = 5.0
ANGLE_STEP_DEG = 10.0
NUM_ANGLES = 36
ANGLE_START = 2


def build_angle_list(random_seed: int) -> List[float]:
    _ = random_seed  # Kept for CLI compatibility; angles are deterministic now.
    return [ANGLE_START_DEG + idx * ANGLE_STEP_DEG for idx in range(NUM_ANGLES)]


def angle_dir_name(angle_deg: float) -> str:
    return f"angle_{angle_deg:07.3f}".replace(".", "p")


def parse_seed_list(seed_spec: str) -> List[int]:
    seeds: List[int] = []
    for part in seed_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid seed range: {part}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def resolve_dataset_path(path_text: str) -> Path:
    dataset_path = Path(path_text)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return dataset_path


def resolve_network_path(path_text: str) -> Path:
    network_path = Path(path_text)
    if network_path.is_absolute():
        return network_path

    explicit_project_path = PROJECT_ROOT / network_path
    if explicit_project_path.is_file() or len(network_path.parts) > 1:
        return explicit_project_path

    matches = sorted(
        TRAINING_RUNS_DIR.rglob(path_text),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if TRAINING_RUNS_DIR.is_dir() else []
    if matches:
        return matches[0]

    return TRAINING_RUNS_DIR / path_text


def label_to_angle_deg(label: List[float]) -> float:
    angle_rad = math.atan2(float(label[ANGLE_START]), float(label[ANGLE_START + 1]))
    return float(math.degrees(angle_rad) % 360.0)


def circular_distance_deg(angle_a: float, angle_b: float) -> float:
    return abs((angle_a - angle_b + 180.0) % 360.0 - 180.0)


def build_condition_label(generator_c_dim: int, class_idx: int, angle_deg: float) -> np.ndarray:
    required_dim = ANGLE_START + 2
    if generator_c_dim < required_dim:
        raise RuntimeError(
            f"Generator c_dim={generator_c_dim}, but this script requires "
            f"labels [T32, T72, sin(angle), cos(angle)] with dim >= {required_dim}."
        )

    label = np.zeros((generator_c_dim,), dtype=np.float32)
    label[int(class_idx)] = 1.0
    angle_rad = math.radians(float(angle_deg))
    label[ANGLE_START] = math.sin(angle_rad)
    label[ANGLE_START + 1] = math.cos(angle_rad)
    return label


def tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu()
    image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image_np = image[0].numpy()
    if image_np.ndim == 3 and image_np.shape[2] == 1:
        return Image.fromarray(image_np[:, :, 0], mode="L")
    return Image.fromarray(image_np, mode="RGB")


def load_generator(network_path: Path, device: torch.device) -> torch.nn.Module:
    with dnnlib.util.open_url(str(network_path), verbose=True) as fp:
        generator = legacy.load_network_pkl(fp)["G_ema"].to(device)
    generator.eval().requires_grad_(False)
    return generator


def generate_images(
    generator: torch.nn.Module,
    device: torch.device,
    seeds: List[int],
    class_idx: int,
    angle_deg: float,
    outdir: Path,
    truncation_psi: float,
    noise_mode: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    label_np = build_condition_label(
        generator_c_dim=int(generator.c_dim),
        class_idx=class_idx,
        angle_deg=angle_deg,
    )
    label = torch.from_numpy(label_np).unsqueeze(0).to(device)

    for seed in seeds:
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, int(generator.z_dim)).astype(np.float32)
        ).to(device)
        with torch.no_grad():
            image_tensor = generator(
                z,
                label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
            )
        image = tensor_to_pil_image(image_tensor)
        image.save(outdir / f"seed{seed:04d}.png")


def load_real_dataset_rows(dataset_path: Path) -> List[Dict[str, object]]:
    with zipfile.ZipFile(dataset_path, "r") as zf:
        metadata = json.loads(zf.read("dataset.json").decode("utf-8"))

    labels = metadata.get("labels")
    if labels is None:
        raise RuntimeError(f"Dataset has no labels: {dataset_path}")

    rows: List[Dict[str, object]] = []
    class_dim = len(CLASS_SPECS)
    for dataset_index, row in enumerate(labels):
        image_name, label = row
        if len(label) < ANGLE_START + 2:
            raise RuntimeError(
                f"Label for {image_name} has dim {len(label)}, expected at least {ANGLE_START + 2}"
            )

        class_idx = max(range(class_dim), key=lambda idx: float(label[idx]))
        rows.append(
            {
                "dataset_index": int(dataset_index),
                "image_name": str(image_name),
                "label": [float(x) for x in label],
                "class_idx": int(class_idx),
                "angle_deg": label_to_angle_deg(label),
            }
        )
    return rows


def select_real_sample(
    rows: List[Dict[str, object]],
    class_idx: int,
    target_angle_deg: float,
) -> Dict[str, object]:
    candidates = [row for row in rows if int(row["class_idx"]) == int(class_idx)]
    if not candidates:
        raise RuntimeError(f"No real samples found for class index {class_idx}")
    return min(
        candidates,
        key=lambda row: circular_distance_deg(float(row["angle_deg"]), float(target_angle_deg)),
    )


def save_real_sample(
    zf: zipfile.ZipFile,
    row: Dict[str, object],
    save_dir: Path,
    class_name: str,
    class_idx: int,
    target_angle_deg: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    image_name = str(row["image_name"])
    real_image_path = save_dir / "real.png"
    real_metadata_path = save_dir / "real_metadata.json"

    with zf.open(image_name, "r") as src:
        real_image_path.write_bytes(src.read())

    matched_angle_deg = float(row["angle_deg"])
    metadata = {
        "class_name": class_name,
        "class_idx": int(class_idx),
        "target_angle_deg": float(target_angle_deg),
        "matched_angle_deg": matched_angle_deg,
        "angle_error_deg": circular_distance_deg(matched_angle_deg, float(target_angle_deg)),
        "dataset_index": int(row["dataset_index"]),
        "archive_image": image_name,
        "label": row["label"],
    }
    real_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T32/T72 images at 8-degree azimuth intervals."
    )
    parser.add_argument(
        "--network",
        default=DEFAULT_NETWORK_NAME,
        help="Network snapshot path. A bare filename is searched recursively under training-runs/.",
    )
    parser.add_argument(
        "--outdir",
        default="generated_t32_t72_angles",
        help="Root output directory.",
    )
    parser.add_argument(
        "--data",
        default=str(DATASET_ZIP),
        help="Dataset zip used to extract nearest real images.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Seed list, e.g. 0, 0-3, or 0,4,8.",
    )
    parser.add_argument(
        "--trunc",
        type=float,
        default=1.0,
        help="Truncation psi.",
    )
    parser.add_argument(
        "--noise-mode",
        choices=["const", "random", "none"],
        default="const",
        help="Noise mode used by the generator.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help="Reserved for compatibility; angles are now fixed at 0, 8, ..., 352 degrees.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    network_path = resolve_network_path(args.network)

    out_root = Path(args.outdir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root

    dataset_path = resolve_dataset_path(args.data)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset zip not found: {dataset_path}")
    if not network_path.is_file():
        message = f"Network snapshot not found: {network_path}"
        if args.dry_run:
            print(f"Warning: {message}")
        else:
            raise FileNotFoundError(message)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Network: {network_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output root: {out_root}")

    real_rows = load_real_dataset_rows(dataset_path)
    seeds = parse_seed_list(args.seeds)

    angles = build_angle_list(args.random_seed)
    print("Angles:")
    for angle in angles:
        print(f"  {angle:.3f} deg")

    generator = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.dry_run:
        print(f"Device: {device}")
        generator = load_generator(network_path=network_path, device=device)
        if int(generator.c_dim) < ANGLE_START + 2:
            raise RuntimeError(
                f"Loaded generator has c_dim={int(generator.c_dim)}. "
                f"Expected c_dim >= {ANGLE_START + 2} for [T32, T72, sin, cos] labels."
            )

    with zipfile.ZipFile(dataset_path, "r") as zf:
        for class_name, class_idx in CLASS_SPECS:
            for angle_deg in angles:
                call_outdir = out_root / class_name / angle_dir_name(angle_deg)
                real_row = select_real_sample(
                    rows=real_rows,
                    class_idx=class_idx,
                    target_angle_deg=angle_deg,
                )
                real_error = circular_distance_deg(float(real_row["angle_deg"]), angle_deg)
                print(
                    f"Real {class_name} target={angle_deg:.3f} deg -> "
                    f"{real_row['image_name']} ({float(real_row['angle_deg']):.3f} deg, "
                    f"err={real_error:.3f} deg)"
                )

                print(
                    f"Generate {class_name} class={class_idx} angle={angle_deg:.3f} deg "
                    f"seeds={seeds} -> {call_outdir}"
                )
                if not args.dry_run:
                    save_real_sample(
                        zf=zf,
                        row=real_row,
                        save_dir=call_outdir,
                        class_name=class_name,
                        class_idx=class_idx,
                        target_angle_deg=angle_deg,
                    )
                    generate_images(
                        generator=generator,
                        device=device,
                        seeds=seeds,
                        class_idx=class_idx,
                        angle_deg=angle_deg,
                        outdir=call_outdir,
                        truncation_psi=float(args.trunc),
                        noise_mode=str(args.noise_mode),
                    )


if __name__ == "__main__":
    main()
