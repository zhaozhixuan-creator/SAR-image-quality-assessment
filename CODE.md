# SAR-IQA · SAR 图像质检引擎

把《SAR 图像质检实现方案》落地为可运行代码：**输入一张现成 SAR 图像，输出 HTML 质检报告 + JSON 结果**。

覆盖方案中的 **7 维 38 项指标** 与 **3 个自研模块**（去斑评价指标包 / RFI 谱域检测器 / 元数据规则引擎）。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`、`Pillow`、`matplotlib`。

## 快速开始

```bash
# 生成仿真样例（可选，验证用）
python examples/generate_samples.py

# 单图质检
python cli.py demo_single.tif

# 带元数据边车 + 标称分辨率
python cli.py demo_quad.tif --metadata meta.json --nominal-resolution 3.0 --pixel-spacing 10.0
```

输出：
- `demo_single_report.html` — 自包含可视化质检报告
- `demo_single_report.json` — 机器可读结果

## 命令行参数

| 参数 | 说明 |
|---|---|
| `image` | 输入图像（PNG/JPG/TIFF，8/16bit 或 float） |
| `--metadata` | 元数据边车 JSON（启用元数据规则引擎） |
| `--domain` | 输入域：`auto`/`amplitude`/`intensity`/`db` |
| `--channels` | 强制通道数解释（默认自动识别单/双/四极化） |
| `--nominal-resolution` | 标称分辨率(m)，用于散焦/展宽因子 |
| `--pixel-spacing` | 像素间距(m)，用于分辨率单位换算 |
| `--level` | 产品级别：`auto`/`L0`/`L1`(SLC)/`L2`/`L3+`（默认 `auto`→L1/SLC） |
| `--batch` | 批量大小，用于 GB/T 24356 抽样建议（≥1001 分批） |
| `--output` / `--json` | 输出路径 |

## 数据集级质检

输入一个 SAR 图像目录（每个图像可选同茎 `.json` 元数据边车），输出数据集整体质量结论，
对应 README §二 的「落库与看板」与「人工抽检」层、及 §七「分组留存率审计」。

```bash
# 生成数据集样例（可选）
python examples/generate_samples.py --dataset

# 数据集级质检
python dataset_cli.py dataset_demo --group-by processor_version
```

输出目录（默认 `<目录>_qc`）：
- `dashboard.html` — 数据集级看板（总体概览 / 评分分布 / 批次判定 / 聚合分布 / 离群 / 元数据一致性 / 分组审计 / 每图明细）
- `dataset_summary.json` — 机器可读汇总
- `records.json` — 逐图全量落库
- `per_image/<stem>.html` / `.json` — 每图报告（复用单图引擎）

四项能力：
1. **聚合统计分布**：跨图汇总每项指标的中位数 / 均值±σ / 分位 / MAD / IQR。
2. **批次合格判定**：按 GB/T 24356 汇总逐图分级为整批合格 / 不合格（任一 A 类缺陷 → 整批不合格），并给出抽样建议。
3. **跨图一致性 / 离群检测**：稳健 MAD / IQR 离群阈值标记异常图像（仅作人工复核，不自动判不合格）；元数据身份键（`processor_version` / `sensor` 等）跨图一致性。
4. **结果落库 + 分组审计**：`--group-by <字段>` 按元数据字段（或产品级）分组统计留存率（非「不合格」即留存）。

主要参数：`--level`、`--domain`、`--channels`、`--recursive`、`--ext`、`--group-by`、
`--batch`（抽样批量）、`--mad-k`（离群阈值）、`--metadata-keys`、`--no-per-image`、`--out-dir`。

## 指标覆盖说明

单张图像能算的指标全部计算并给出数值；需角反射器 / 时序 / 干涉对 / 原始回波的指标
在报告中明确标注「无法评估」并说明原因。

- **直接计算**（约 20 项）：饱和率、扇贝、子带台阶、EAP 残差、负值率、丢行/坏像元、
  ENL 双估计量、分辨率/PSLR/ISLR/SSLR（自动检测图像内强点目标）、通道幅度不平衡、
  互易性 VH≈HV、通道间配准、有效覆盖率、无效值等。
- **代理/有条件**：NESZ（暗区噪声底代理）、RFI（图像域条纹代理 + 谱域接口）、
  去斑评价（内部 Lee 滤波作参考）、元数据规则（有边车则跑）。
- **明确标注需外部数据**：ALE、绝对定标精度、稳定性、串扰、AASR/RASR、
  干涉相干性等 5 项、CARD4L 合规（无元数据时）等。

## 方案落地补充（README §二 / §四 / §五 / §八）

除单图筛查外，引擎把方案中其余四块落为结构化代码，全部由 `spec.py` 一份事实源驱动：

- **分级质检流水线**（§二）：`pipeline.py` 按产品级（L0 / L1-SLC / L2 / L3+）组织 38 项，
  `--level` 指定当前产品级；非原生级指标的测得值标注「跨级测量告警」。
- **判据与缺陷分级**（§五）：`grading.py` 在原有超差比例表 + 百分制扣分之上，补齐
  GB/T 24356 抽样量表（`--batch` 出抽样建议）与**判决项 / 标记项分离**（RFI、叠掩阴影、
  低相干区为标记项，只作可用性元数据、不计分）。
- **开源生态选型**（§八）：`ecosystem.py` 登记约 25 个参考实现的许可证与角色，报告
  按许可类别（可集成 / GPL 只宜研读 / 需核对 / 非商业 / 数据集）输出合规结论。
- **分阶段落地路径**（§四）：`spec.py` 为每项打 `phase` 标签（一期前置筛查 / 二期 sct /
  三期自研+元数据），报告与 JSON 均展示各期指标数。

每项指标在报告与 JSON 中附带 `level` / `phase` / `kind` / `refs`，实现「每个数值可追溯」。

## 目录结构

```
cli.py                    单图命令行入口
dataset_cli.py            数据集命令行入口（目录扫描 → 看板 + 逐图报告）
sar_iqa/
  io.py                   读图与归一化
  base.py                 MetricResult / Status / 注册表
  spec.py                 规格单一事实源：38 项指标的 level/phase/kind/method/refs
  ecosystem.py            开源生态选型 + 许可合规注册表
  pipeline.py             分级质检流水线（L0/L1-SLC/L2/L3+ 级别门控与汇总）
  profiles.py             行/列剖面、去趋势、周期/阶跃检测
  point_target.py         点目标检测 + IRF 过采样分析
  grading.py              超差比例 → A/B/C/D → 百分制 + 抽样量表 + 判决/标记分离
  report.py               HTML 报告（含分级/标记/抽样/生态小节）
  plots.py                matplotlib → base64
  single.py               单图评估单元（assess_image / 去斑模块 / JSON 序列化）
  dataset.py              数据集级库（扫描 / 聚合 / 批次判定 / 离群 / 分组审计 / 落库）
  dataset_report.py       数据集级看板 HTML
  metrics/                7 个维度 38 项指标
  modules/                去斑评价 / RFI / 元数据规则
```

## 重要说明

- 本引擎是 **单图筛查**，结果不能替代完整验收流水线（需角反射器、时序、干涉对、
  原始回波才能覆盖全部 38 项）。
- 报告中的评分采用 GB/T 24356 的「超差比例 → A/B/C/D → 百分制」口径，但阈值基于
  方案文档中的公开基准，实际生产需按处理器/传感器版本绑定定制阈值。
