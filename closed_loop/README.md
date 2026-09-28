# SAR 生成图像质检「闭环反馈」系统

把 **SAR-Generation 的 agent 闭环骨架**（`RequirementParser → Router → Planner → Executor → FeedbackOptimizer`）
与 **v3 质检引擎**（3 模型 + 13 指标）结合：质检不再只给分数，而是据此决定「下一步该动什么」——
**重选模型 / 调整参数 / 重训**，并通过「反馈后指标是否改善」反向验证质检的有效性（专利核心新颖性）。

## 闭环流程

```
自然语言需求 ─► 解析(requirement_parser) ─► 路由(router: task→模型/变体)
   ─► 评估(复用 v3 13 指标) ─► 弱点检测(portrait) ─► 反馈(feedback: 三选一)
   ─► 复评验证(指标改善?) ─► 输出结构化画像 portrait.json
```

## 目录结构

```text
closed_loop/
├── configs/          system/datasets/models 三份配置（models 登记 3 模型能力边界）
├── agent/            requirement_parser / router / planner / executor / feedback_optimizer
├── feedback/         reselect(重选) / adjust_param(调参) / retrain(重训) 三机制
├── evaluation/       metrics_bridge(复用 v3) / portrait(画像+弱点检测)
├── models/           registry（能力元数据 + v3 生成入口）
├── tools/            io_utils / package_utils（自 SAR-Generation 拷贝）
├── scripts/          run_closed_loop.py(主入口) / run_feedback_experiments.py(三实验)
└── results/          experiments/ 证据 + logs/ 日志（备填专利模板）
```

## 快速开始

```bash
cd closed_loop

# 1) 只解析+路由+规划（不评估）
python3 scripts/run_closed_loop.py \
  --request "给 5 类各生成 100 张 128×128，方位角均衡" --dry-run

# 2) 完整闭环（复用 v3 既有评估结果，纯逻辑、零 GPU）
python3 scripts/run_closed_loop.py \
  --request "参考图换视角，要求背景统计一致"

# 3) 四个「质检有效性」验证实验（exp2/exp3/exp4 需 GPU）
python3 scripts/run_feedback_experiments.py --exp 1     # 重选（免费）
python3 scripts/run_feedback_experiments.py --exp 2     # 调参（angle_gen 扩散步数）
python3 scripts/run_feedback_experiments.py --exp 3     # 重训（stylegan 朴素续训）
# exp4（方位角定向微调，在 v3 评估 venv 内运行）：
/home/zhaozhixuan/SAR-Generation/sar_models/GenModel4SAR/angle_gen/.venv/bin/python \
  scripts/run_azimuth_finetune.py --gpu 2 --kimg 200 --az-weight 1.0
```

> 说明：闭环默认运行在「复用模式」——读取 `v3/results/metrics/*/*.json` 的既有指标完成
> 反馈与验证，不消耗 GPU。全量生成模式由 `scripts/run_feedback_experiments.py` 编排
> （分别调用 v3 各模型 venv 内的生成/训练脚本）。

## 结构化 I/O

**输入**（`requirement_parser` 输出，`task_schema`）：

```json
{
  "task_type": "batch_generation | novel_view_synthesis | cross_band_translation",
  "target_classes": ["2S1"], "generation_count": 100, "image_size": [128, 128],
  "azimuth_requirement": "balanced", "reference": null,
  "quality_requirements": {"background_consistency": "high", "structure_fidelity": "high", "azimuth_precision": "low"}
}
```

**输出**（`portrait.json`，机器可读，供下游需求解析侧消费）：

```json
{
  "selected_model": "angle_gen", "selected_variant": "baseline",
  "verdict": "weak(background)", "feedback": {"action": "reselect", ...},
  "improvement": {"metric": "delta_enl", "from_value": 2.374, "to_value": 0.383, "relative_gap": 0.84}
}
```

## 三反馈机制

| 机制 | 文件 | 触发条件 | 验证方式 |
|---|---|---|---|
| 重选模型 | `feedback/reselect.py` | 背景统计类弱点（ΔENL/BVE）且存在更优变体 | 改派后指标改善（exp1：ΔENL 2.37→0.38，−84%） |
| 调整参数 | `feedback/adjust_param.py` | 单模型内超参数可扫（扩散步数） | 指标随参数单调变化（exp2：SSIM/PSNR 随去噪步数↑） |
| 重训练 | `feedback/retrain.py` | 结构保真/角度控制等能力性弱点 | 续训后复评指标改善 |

**重训机制的两种实现**（`feedback/retrain.py` 的 `azimuth_weight` 参数切换）：

| 模式 | 说明 | 结果 |
|---|---|---|
| 朴素续训 | 直接 `--resume` 续训同类数据 | exp3：中性（200 kimg 无显著改善） |
| **方位角定向微调** | 把质检系统的辅助方位角估计器 R(·)（算 CMAE 的网络）作为**冻结教师**，加方位角一致性损失 | exp4：CMAE 显著下降 |

> 「方位角定向微调」是『质检不但说哪里差、还指导往哪改』的**核心落地**：质检画像定位到
> CMAE≈91°（方位角控制≈随机）的弱点后，不再盲目续训，而是把质检自己的估计器 R(·) 反过来
> 作为可微教师监督生成器，强制生成图朝向匹配条件方位角。实现位于
> `sar_models/stylegan4SAR/plugin/azimuth_loss.py` + `training/loss.py`（Gmain 增项）。

## 四实验证据（备填专利模板）

| 实验 | 结果文件 | 机制 |
|---|---|---|
| exp1 | `results/experiments/exp1_reselect.json` | 重选 |
| exp2 | `results/experiments/exp2_adjust_param.json` | 调参 |
| exp3 | `results/experiments/exp3_retrain.json` | 重训（朴素续训） |
| exp4 | `results/experiments/exp4_azimuth_retrain.json` | 重训（方位角定向微调） |

每个实验的生成 manifest、对齐 `.npy`、指标 JSON、改进幅度 delta、运行日志均落盘
`results/` 下，作为撰写专利时的**工作痕迹**逐字段回填。
