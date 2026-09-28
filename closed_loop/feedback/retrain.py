"""反馈机制 3 —— 重训练。

当质检判定持续性弱点（如 stylegan 结构保真弱 SSIM/AFS、或方位角控制弱 CMAE）时，
触发生成模型重训/微调：从当前 checkpoint 续训，产出新权重后重新评估对比。

本模块只负责「构造重训命令与元数据」；实际训练由 scripts/run_feedback_experiments.py
在对应 venv 内执行（GPU）。stylegan 续训参数来源于其 training_options.json + train.py
（--resume 支持从快照续训）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_stylegan_retrain_command(model_id: str, weakness_metrics: set[str],
                                   checkpoint: str, data_zip: str, outdir: str,
                                   kimg: int = 200, gpu: int = 2,
                                   azimuth_weight: float = 0.0,
                                   azimuth_estimator: str = "") -> dict[str, Any]:
    """构造 stylegan 续训命令（供实验脚本在 stylegan venv 内执行）。

    azimuth_weight > 0 时启用「方位角定向重训」：把质检系统的辅助方位角估计器 R(·)
    （算 CMAE 的网络）作为冻结教师，加方位角一致性损失，专治 CMAE 弱点。
    """
    reason = {
        "ssim": "结构保真度低",
        "afs": "目标区散射结构偏差",
        "cmae_deg": "方位角控制弱",
    }
    triggers = [reason[m] for m in weakness_metrics if m in reason] or ["指标偏弱"]

    cmd = [
        "python", "train.py",
        f"--outdir={outdir}",
        f"--data={data_zip}",
        "--gpus=1",
        "--cond=1",
        "--cfg=auto",
        "--aug=ada",
        "--batch=32",
        f"--kimg={kimg}",
        "--snap=25",
        "--seed=0",
        "--metrics=none",
        "--workers=3",
        f"--resume={checkpoint}",
    ]
    method = "stylegan2-ada resume fine-tune"
    if azimuth_weight > 0:
        cmd += [f"--azimuth_weight={azimuth_weight}", f"--azimuth_estimator={azimuth_estimator}"]
        method = "stylegan2-ada 方位角定向微调（质检估计器 R 作为冻结教师）"
    return {
        "action": "retrain",
        "model_id": model_id,
        "trigger_weakness": sorted(weakness_metrics),
        "reason": "、".join(triggers),
        "method": method,
        "resume_checkpoint": checkpoint,
        "data": data_zip,
        "kimg": kimg,
        "gpu": gpu,
        "azimuth_weight": azimuth_weight,
        "azimuth_estimator": azimuth_estimator,
        "command": cmd,
    }


def build_retrain_action(model_id: str, weakness_metrics: set[str],
                         v3_model_cfg: dict[str, Any], variant: str = "stylegan2_ada_5class",
                         outdir: str = "results/retrain/stylegan") -> dict[str, Any]:
    """从 v3 模型配置 + 弱点构造重训动作（stylegan 专用）。

    v3_model_cfg = v3 config.yaml 中 models.stylegan4sar 那一节（含 checkpoint/venv/cwd）。
    """
    if model_id != "stylegan4sar":
        return {"action": "retrain", "applied": False,
                "reason": f"当前仅实现 stylegan4sar 的重训通道，{model_id} 需另行实现"}

    variant_cfg = v3_model_cfg["variants"].get(variant, {})
    checkpoint = variant_cfg.get("checkpoint", "")
    # 训练数据 zip 与 checkpoint 同仓（相对 v3 配置 root 解析，这里透传绝对路径由实验脚本解析）
    data_zip = variant_cfg.get("train_data", "datasets/mstar15_5class_stride5_maxdev-128.zip")
    return build_stylegan_retrain_command(
        model_id, weakness_metrics, checkpoint, data_zip, outdir)
