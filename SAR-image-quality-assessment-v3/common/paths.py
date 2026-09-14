"""路径解析：配置加载、{root} 占位符展开、远程机绝对路径重映射。

v3 与运行时仓库 SAR-Generation 分离；manifest/meta.pkl 内硬编码了远程机
绝对路径 ``/data/zqm/SAR-Generation``，读取时需统一重映射到本机 ``root``。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

V3_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """加载 config.yaml 并返回 dict。"""
    path = Path(path) if path else V3_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(value, cfg: dict) -> str:
    """把配置值里的 ``{root}`` 等占位符替换为实际路径。

    相对路径（以 ``../`` 或 ``./`` 开头）相对于 v3 根目录解析。
    """
    if not isinstance(value, str):
        return value
    value = value.replace("{root}", str(cfg.get("root", "")))
    if value.startswith("./") or value.startswith("../"):
        value = str((V3_ROOT / value).resolve())
    return value


def remap_remote(path: str, cfg: dict) -> str:
    """把远程机绝对路径重映射到本机 root。"""
    remote = str(cfg.get("remote_root", "/data/zqm/SAR-Generation"))
    local = str(cfg.get("root", ""))
    if path.startswith(remote):
        return local + path[len(remote):]
    return path


def _abs(base: str, cfg: dict) -> Path:
    p = Path(resolve(base, cfg))
    if not p.is_absolute():
        p = V3_ROOT / p
    return p


def ws_dir(cfg: dict, *parts) -> Path:
    """workspace 目录下的子路径（绝对路径）。"""
    return _abs(cfg.get("workspace_dir", "workspace"), cfg).joinpath(*parts)


def res_dir(cfg: dict, *parts) -> Path:
    """results 目录下的子路径（绝对路径）。"""
    return _abs(cfg.get("results_dir", "results"), cfg).joinpath(*parts)


def resolve_device(cfg: dict) -> str:
    """解析评估设备：裸 ``cuda`` 补上 GPU 编号；``cuda:N``/``cpu`` 原样返回。

    兼容 ``--device cuda:3`` 覆盖（若写成 cuda:* 会被强制改写，覆盖会静默失效）。
    """
    d = str(cfg.get("evaluation", {}).get("device", "cpu"))
    if d.startswith("cuda") and ":" not in d:
        d = f"cuda:{cfg['evaluation']['gpu']}"
    return d
