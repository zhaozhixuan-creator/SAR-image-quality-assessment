# 工作流契约（WORKFLOW）

本文件是 v3 流水线的**阶段契约**：每个阶段（stage）是一个独立模块，仅通过磁盘产物
（`workspace/` 与 `results/`）解耦，阶段之间不共享进程内状态。因此每个阶段可以 1:1
替换为独立 agent / subprocess，而不改动相互接口。

## 阶段注册表

编排器 `run_pipeline.py` 里有一个显式的 `STAGES` 字典（`prepare → generate → align →
evaluate → report`），不再用 `importlib.import_module(f"pipeline.stage_{name}")` 字符串拼接。

```python
# run_pipeline.py
STAGES = {
    "prepare": run_prepare,    # pipeline/stage_prepare.py
    "generate": run_generate,  # pipeline/stage_generate.py
    "align": run_align,        # pipeline/stage_align.py
    "evaluate": run_evaluate,  # pipeline/stage_evaluate.py
    "report": run_report,      # pipeline/stage_report.py
}
```

每个 stage 暴露统一入口 `run(cfg: dict, args: argparse.Namespace) -> None`，`cfg` 来自
`config.yaml`，`args` 来自 CLI（`--stage/--model/--device/--gpu`）。`args` 可传 `None`。

## 阶段 consume → produce 契约

| 阶段 | 消费（输入） | 产出（输出） |
|---|---|---|
| **prepare** | `config.datasets.image_root` 真实 `.jpg`；`metadata_{train,test}` 的 `meta.pkl` | `workspace/real/{size}/{split}/{images.npy,labels.json,summary.json}` |
| **generate** | `workspace/real/.../labels.json`（stylegan 条件）；各模型 checkpoint/venv | `workspace/generated/{model}/{variant}/{*.npy,generation_manifest.json}` |
| **align** | `workspace/real/.../images.npy`（训辅助网络）；`workspace/generated/.../generation_manifest.json` | `workspace/aligned/{model}/{variant}/{real.npy,fake.npy,angles.npy?}` + `workspace/checkpoints/{R_128.pt,R_64.pt,E_asc.pt}` |
| **evaluate** | `workspace/aligned/.../{real,fake,angles}.npy` + `workspace/checkpoints/*.pt` | `results/metrics/{model}/{variant}.json`（6+7 项指标，不适用项为 `null`） |
| **report** | `results/metrics/**/*.json` | `results/{summary.json,report.md,report.html}` |

## 关键不变量

- **数据口径**：所有进入指标的图统一为 `list[np.ndarray]` 幅度域 `[0,1]` 灰度图，逐对对齐。
  ΔENL/BVE 在强度域（幅度²）计算（论文式 22/41/42）。
- **指标适用矩阵**：`config.yaml` 每模型 `metrics` 字段声明适用子集；`metrics_bridge` 对未声明
  项返回 `None`，报告 schema 稳定（显 `—`）。未声明 `metrics` 的模型默认全 13 项。
- **FR 配对口径**：七项 FR 依赖「真实↔生成」如何配对，各模型 `pairing` 字段声明语义
  （stylegan=跨样本同类；angle_gen=真·新视角；frequency=翻译；gaussrecon=重构）。跨模型比较
  FR 前需先对齐口径。
- **辅助网络缓存**：`stage_align` 若检测到 `R_128.pt / R_64.pt / E_asc.pt` 已存在则跳过训练；
  删除对应 `.pt` 可强制重训（真实集未变时结果可复现）。
- **设备解析**：`common.paths.resolve_device(cfg)` 统一解析——裸 `cuda` 补 GPU 编号，`cuda:N`
  原样保留（兼容 `--device cuda:3` 覆盖）。

## 映射到 multi-agent

- 每个 stage 是未来的一个 agent seam。要把它改成独立进程/agent，只需：
  1. 序列化 `cfg`（已是 YAML）与 `args` 传给子进程；
  2. 子进程内 `paths.load_config()` 重建 `cfg`，调用对应 `run(cfg, args)`；
  3. 产物的磁盘路径不变，上下游阶段无需改动。
- 生成器（`generators/*.py`）已按子进程方式运行（各自 `.venv` 内 `subprocess.run`），是这一
  模式的既有先例。
- 报告侧特征总结（擅长/不擅长）目前是启发式阈值（`stage_report._characterize`）+ 组内相对最弱
  top-K 标注；若要更强的「模型特征总结」，可在该 stage 内替换为 agent 综述，而不影响其他阶段。

## 单阶段运行

```bash
python run_pipeline.py --stage prepare
python run_pipeline.py --stage generate --model angle_gen
python run_pipeline.py --stage evaluate --device cuda:3
```
