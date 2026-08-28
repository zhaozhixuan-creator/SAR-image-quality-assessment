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
| `--output` / `--json` | 输出路径 |

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

## 目录结构

```
cli.py                    命令行入口
sar_iqa/
  io.py                   读图与归一化
  base.py                 MetricResult / Status / 注册表
  profiles.py             行/列剖面、去趋势、周期/阶跃检测
  point_target.py         点目标检测 + IRF 过采样分析
  grading.py              超差比例 → A/B/C/D → 百分制 → 优/良/合格/不合格
  report.py               HTML 报告
  plots.py                matplotlib → base64
  metrics/                7 个维度 38 项指标
  modules/                去斑评价 / RFI / 元数据规则
```

## 重要说明

- 本引擎是 **单图筛查**，结果不能替代完整验收流水线（需角反射器、时序、干涉对、
  原始回波才能覆盖全部 38 项）。
- 报告中的评分采用 GB/T 24356 的「超差比例 → A/B/C/D → 百分制」口径，但阈值基于
  方案文档中的公开基准，实际生产需按处理器/传感器版本绑定定制阈值。
