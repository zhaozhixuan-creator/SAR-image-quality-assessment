# SAR 生成图像质检评估工作流（v3）

把 **v2 评估端**（论文 §3.1.4 六项生成指标 + 七项 FR 全参考指标）桥接到 **3 个 SAR 生成模型（4 个子目录）**，做生成图像的质检评估，并用评估结果总结每个模型的特征（擅长/不擅长）。

## 设计原则

- **模块化流水线**：config 驱动、阶段清晰分离、阶段间仅通过磁盘 JSON 契约解耦。每个阶段是独立模块，后续可 1:1 映射为 multi-agent，无需改动相互接口。
- **最小 skill 痕迹**：本目录是普通 Python 项目（config + CLI 编排器 + 模块 + README），不含任何 SKILL.md / 斜杠命令。
- **不复制 v2 实现**：指标引擎复用 `SAR-image-quality-assessment-v2/gan_metrics`（`evaluation/metrics_bridge.py` 通过 `sys.path` 导入）。

## 目录结构

```text
SAR-image-quality-assessment-v3/
├── config.yaml              # 全局配置：路径重映射、模型/数据集/指标注册、各模型 venv、流水线开关
├── run_pipeline.py          # 顶层编排器（--stage / --model / --device，内含 STAGES 注册表）
├── WORKFLOW.md              # 阶段 consume→produce 契约（multi-agent 映射说明）
├── common/                  # 路径解析、远程路径重映射、设备解析、图像读写
├── pipeline/                # stage_{prepare,generate,align,evaluate,report}.py（= 未来 agent seam）
├── generators/              # 各模型推理封装（在各自 .venv 内被调用）
├── evaluation/              # metrics_bridge.py（复用 v2 gan_metrics）
├── workspace/               # 运行时产物（real/generated/aligned/checkpoints），gitignore
└── results/                 # 指标 JSON + 报告（HTML/MD/summary.json），gitignore
```

## 快速开始

```bash
cd SAR-image-quality-assessment-v3

# 全量流水线（prepare → generate → align → evaluate → report）
python run_pipeline.py

# 或分阶段
python run_pipeline.py --stage prepare            # A 数据准备
python run_pipeline.py --stage generate           # B 生成（4 子目录）
python run_pipeline.py --stage align              # C 对齐 + 训练 R/E_asc
python run_pipeline.py --stage evaluate           # D 评估（6+7 指标）
python run_pipeline.py --stage report             # E 报告
```

> 说明：`generate`/`align`/`evaluate` 需要在对应环境运行（见下）。评估环境为 `config.yaml` 中
> `evaluation.python`（angle_gen `.venv`，py3.10/torch2.4/cuda，已装 scipy/skimage/piq/pytorch-msssim）。
> 生成阶段由编排器按 `config.yaml` 中每个模型的 `venv_python` 自动切换到各自 `.venv`。

## 指标适用矩阵

各模型适用哪些指标由 `config.yaml` 中该模型的 `metrics` 字段声明（`metrics_bridge` 对未声明的项返回 `None`）：

| 指标 | stylegan4SAR | angle_gen | frequency_gen | gaussrecon4sar |
|---|---|---|---|---|
| FID / ΔENL / BVE（分布级） | ✅ 两变体 | ✅ 两变体 | ❌（无真实分布） | ✅ 三变体 |
| SSIM（结构，配对） | ✅ | ✅ | ✅（99 对） | ✅ |
| AFS（目标区 ASC 结构） | ✅ | ✅ | ❌（域不匹配） | ✅ |
| CMAE（角度条件） | ✅（条件角） | ✅（目标角） | ❌（无角度） | ✅（文件名方位角） |
| 7 项 FR（全参考配对） | ✅ | ✅ | ✅（99 对） | ✅ |

## 覆盖的生成模型（4 子目录）

| 子目录 | 模型 | 变体 | 状态 |
|---|---|---|---|
| `stylegan4SAR` | 条件 StyleGAN2-ADA | 基线 / 增强(AASG+ESF+ENL) | 10 类已训 5000kimg，✅ 全量评估 |
| `GenModel4SAR/angle_gen` | XUNet 扩散 NVS | geometry / baseline / adapted 姿态编码 | ✅ 全量评估（64×64） |
| `GenModel4SAR/frequency_gen` | X→Ka Pix2Pix | — | ✅ 真实 wholeImg 测试集 99 对（SSIM+FR） |
| `GaussRecon4SAR` | 3DGS 电磁反演/重构 | image_SSIM / image_SSIM_sanshezhongxin / reconstruction_CD_sanshezhongxin | ✅ 预生成 renders/gt 评估（本机 Linux 无法重生成） |

> 说明：
> - **angle_gen `adapted`**：与 `geometry` 同构（XUNet 1 通道）的另一份权重，是仓库默认的
>   `pretrain` 源（`options.py` 未显式给 `pretrain_model_pth` 时默认 `models/latest_adapted.pt`，
>   step 101715，低于 `SAR_geometry.pt` 的 690000）。它**不是第三个独立架构**，加进来是为单独
>   分离「adaptation / checkpoint 选择」这一个变量做对照，个别指标可能不如全量 geometry。
> - **GaussRecon4SAR 的另 4 个实验未纳入**：`image_SSIM`（ours_30000）、`NVS`、`reconstruction_CD`
>   （ours_300000）、`SSIM_CONFIG_TEST` 在预生成目录里为空/中止（0~1 张），无法配对评估，
>   故已排除。若后续补充 renders/gt，可在 `config.yaml` 的 `gaussrecon4sar.variants` 加对应 `exp`。

## 关键路径（在 `config.yaml` 的 `root`，默认 `/home/zhaozhixuan/SAR-Generation`）

- 真实参考：`data/raw/MSTAR/soc/{train,test}/{10类}/*.jpg`（158×158 幅度图）+ `angle_gen/data/MSTAR_SOC_10CLASS/{train,test}/meta.pkl`（方位角/入射角）。
- StyleGAN 10 类 checkpoint：`stylegan4SAR/training-runs/mstar_soc10_10class/{stylegan2_ada_v1,enhanced_stylegan4sar_v1}/.../network-snapshot-005000.pkl`。
- angle_gen checkpoint：`GenModel4SAR/angle_gen/experiments/diffusion_ori/pretrained/models/{SAR_geometry.pt,baseline.pt,latest_adapted.pt}`。
- 远程路径 `/data/zqm/SAR-Generation` 由 `common/paths.remap_remote` 重映射到本机 `root`。

## 指标口径（复用 v2）

- 输入统一为 `list[np.ndarray]` 幅度域 [0,1] 灰度图。
- ΔENL / BVE 在强度域（幅度²）计算（论文式 22/41/42）。
- R(·) 方位角估计器在真实集上预训练、评估时冻结；E_asc 以重构自监督预训练（论文口径的简化替身）。
- FID 用 torchvision Inception-v3（离线时回退随机初始化，仅流程演示）。
