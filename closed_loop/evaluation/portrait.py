"""模型画像 + 弱点检测（从 v3 pipeline/stage_report.py 提炼，纯逻辑、无 torch 依赖）。

输入一组实测指标 → 输出结构化画像：擅长/不擅长、弱点指标集合、裁决 verdict。
反馈机制据此决定动作（重选 / 调参 / 重训）。

指标方向与绝对阈值口径与 v3 stage_report 保持一致，保证画像与 v3 报告同源。
"""
from __future__ import annotations

from typing import Any

PAPER_KEYS = ["fid", "ssim", "afs", "delta_enl", "bve", "cmae_deg"]
FR_KEYS = ["mse", "rmse", "psnr", "ncc", "uqi", "ms_ssim", "fsim"]
ALL_KEYS = PAPER_KEYS + FR_KEYS

# +1 越大越好，-1 越小越好
IDEAL = {
    "fid": -1, "ssim": +1, "afs": +1, "delta_enl": -1, "bve": -1, "cmae_deg": -1,
    "mse": -1, "rmse": -1, "psnr": +1, "ncc": +1, "uqi": +1, "ms_ssim": +1, "fsim": +1,
}

# 弱点指标 → 可采取的反馈动作（机制选择提示）
WEAK_METRIC_ACTION = {
    "ssim": "retrain",        # 结构保真弱 → 重训（或重选变体）
    "afs": "retrain",         # 目标区散射结构偏差 → 重训
    "delta_enl": "reselect",  # 背景噪声统计偏差 → 优先重选更稳变体
    "bve": "reselect",
    "ncc": "retrain",
    "cmae_deg": "retrain",    # 角度控制弱 → 重训（当前无模型达标）
}


def characterize_metrics(m: dict[str, Any]) -> tuple[list[str], list[str], set[str]]:
    """绝对阈值 → (strengths, weaknesses, weak_metrics)。与 v3 stage_report 同口径。"""
    strengths, weaknesses = [], []
    weak_metrics: set[str] = set()

    if m.get("ssim") is not None:
        if m["ssim"] >= 0.6:
            strengths.append("结构保真度高（SSIM）")
        elif m["ssim"] < 0.35:
            weaknesses.append("结构保真度低（SSIM）")
            weak_metrics.add("ssim")
    if m.get("afs") is not None:
        if m["afs"] >= 0.9:
            strengths.append("目标区散射结构一致（AFS）")
        elif m["afs"] < 0.7:
            weaknesses.append("目标区散射结构偏差（AFS）")
            weak_metrics.add("afs")
    if m.get("delta_enl") is not None:
        if m["delta_enl"] < 0.5:
            strengths.append("背景统计一致性（ΔENL）")
        elif m["delta_enl"] > 2.0:
            weaknesses.append("背景噪声统计偏差（ΔENL）")
            weak_metrics.add("delta_enl")
    if m.get("bve") is not None and m["bve"] > 0.05:
        weaknesses.append("背景方差误差偏大（BVE）")
        weak_metrics.add("bve")
    if m.get("cmae_deg") is not None and m["cmae_deg"] >= 45:
        weaknesses.append("方位角控制弱（CMAE）")
        weak_metrics.add("cmae_deg")
    if m.get("ncc") is not None:
        if m["ncc"] >= 0.5:
            strengths.append("与真实图相关性高（NCC）")
        elif m["ncc"] < 0.2:
            weaknesses.append("与真实图相关性低（NCC）")
            weak_metrics.add("ncc")

    return strengths, weaknesses, weak_metrics


def verdict(m: dict[str, Any], weak_metrics: set[str]) -> str:
    """根据弱点指标集合给出裁决。"""
    if not weak_metrics:
        return "pass"
    primary = sorted(weak_metrics, key=lambda k: IDEAL.get(k, 0))[0] if weak_metrics else ""
    return f"weak({primary})"


def suggest_action(weak_metrics: set[str]) -> str:
    """弱点 → 首选反馈动作。"""
    if not weak_metrics:
        return "none"
    # 有重选可解决的背景统计问题优先重选；否则重训
    if weak_metrics & {"delta_enl", "bve"}:
        return "reselect"
    return "retrain"


def build_portrait(model_id: str, variant: str, metrics: dict[str, Any],
                   request: dict[str, Any] | None = None,
                   candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """把实测指标 → 结构化画像（portrait.json 的单一模型块）。"""
    strengths, weaknesses, weak_metrics = characterize_metrics(metrics)
    return {
        "model": model_id,
        "variant": variant,
        "metrics": {k: metrics.get(k) for k in ALL_KEYS},
        "characterization": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "weak_metrics": sorted(weak_metrics),
        },
        "verdict": verdict(metrics, weak_metrics),
        "suggested_action": suggest_action(weak_metrics),
        "request": request,
        "candidates": candidates,
    }
