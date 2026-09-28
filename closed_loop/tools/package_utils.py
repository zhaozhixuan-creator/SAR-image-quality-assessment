from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tools.io_utils import ensure_dir, list_images, resolve_path, write_json


def create_dataset_package(task_schema: dict[str, Any], generated_dir: str | Path, package_root: str | Path) -> dict[str, Any]:
    task_id = task_schema["task_id"]
    source_dir = resolve_path(generated_dir)
    package_dir = ensure_dir(Path(package_root) / task_id)
    images_dir = ensure_dir(package_dir / "images")

    copied = []
    for image_path in list_images(source_dir):
        target = images_dir / image_path.name
        shutil.copy2(image_path, target)
        copied.append(str(target))

    label_path = source_dir / "labels.csv"
    packaged_label_path = None
    if label_path.exists():
        packaged_label_path = package_dir / "labels.csv"
        shutil.copy2(label_path, packaged_label_path)

    manifest = {
        "task_id": task_id,
        "dataset": task_schema.get("dataset"),
        "target_classes": task_schema.get("target_classes", []),
        "image_count": len(copied),
        "images": copied,
        "label_file": str(packaged_label_path) if packaged_label_path else None,
    }
    manifest_path = write_json(package_dir / "manifest.json", manifest)
    return {
        "package_dir": str(package_dir),
        "manifest_path": str(manifest_path),
        "label_file": str(packaged_label_path) if packaged_label_path else None,
        "image_count": len(copied),
    }
