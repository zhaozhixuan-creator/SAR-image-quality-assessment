from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def ensure_dir(path: str | Path) -> Path:
    value = resolve_path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    value = resolve_path(path)
    text = value.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(f"Cannot parse {value}; install PyYAML or keep config as JSON-compatible YAML.") from exc
        data = yaml.safe_load(text)
        return data or {}


def write_json(path: str | Path, data: Any) -> Path:
    value = resolve_path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    value.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def write_text(path: str | Path, text: str) -> Path:
    value = resolve_path(path)
    value.parent.mkdir(parents=True, exist_ok=True)
    value.write_text(text, encoding="utf-8")
    return value


def new_task_id(prefix: str = "sar") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def list_images(root: str | Path) -> list[Path]:
    value = resolve_path(root)
    if not value.exists():
        return []
    return sorted(path for path in value.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_file(src: str | Path, dst: str | Path) -> Path:
    source = resolve_path(src)
    target = resolve_path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
