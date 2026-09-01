# SAR-image-quality-assessment-v2

基于论文 *Remote Sensing* **remotesensing-18-02247**（SAR 未观测方位角图像生成，StyleGAN2 主干）
§3.1.4 "Evaluation Metrics" 的**生成模型评估指标**实现与验证。

> 与 [../SAR-image-quality-assessment-v1](../SAR-image-quality-assessment-v1)（单图质检引擎）相互独立：
> v2 评估的是「生成图像集」与「真实图像集」在六项指标上的**分布一致性**，用于给生成模型打分，
> 而非对单张 SAR 图像做 7 维 38 项质检。

验证数据支持两种：**真实 MSTAR**（默认）与**仿真 MSTAR-like**（替身）。

---

## 一、六项指标

| 指标 | 论文公式 | 度量对象 | 理想值（生成≈真实） |
|---|---|---|---|
| **FID** | ‖μr−μg‖² + Tr(Σr+Σg−2(ΣrΣg)^{1/2}) | Inception-v3 特征分布距离 | → 0 |
| **SSIM** | 结构相似度 | 逐像素结构保真 | → 1 |
| **AFS** | (1/N)Σ cos(E_asc(x_t), E_asc(x̂_t)) | 目标区 ASC 感知特征相似度 | → 1 |
| **ΔENL** | (1/N)Σ\|ENL(x_b)−ENL(x̂_b)\| | 背景等效视数误差 | → 0 |
| **BVE** | (1/N)Σ\|σ²(x_b)−σ²(x̂_b)\| | 背景方差误差 | → 0 |
| **CMAE** | (1/N)Σ min(\|φ̂−φ\|, 360−\|φ̂−φ\|) | 方位角循环平均绝对误差 | → 0 |

其中 ENL = μ²/(σ²+ε)，ε=1e-6；φ̂ 由预训练方位角估计器 R(·) 反推。

## 二、目录结构

```
SAR-image-quality-assessment-v2/
├── gan_metrics/                    # 核心库：六项指标 + 两个辅助网络 + Inception 特征
│   ├── __init__.py
│   ├── common.py                   # 目标/背景掩码、灰度→RGB、目标区裁剪
│   ├── inception.py                # Inception-v3 特征提取（FID）
│   ├── models.py                   # AspectAngleEstimator(R) / ASCExtractor(E_asc)
│   └── metrics.py                  # fid/ssim/afs/delta_enl/bve/cmae + enl/estimate_angles
├── examples/
│   ├── download_mstar.py           # 下载真实 MSTAR .npz（分片并行，国内可用）
│   ├── mstar_to_chips.py           # 真实 MSTAR → v2 切片口径（mstar_real/）
│   ├── inspect_mstar.py            # 核查 .npz 类名/角度/俯角分布
│   ├── generate_mstar_like.py      # 生成仿真 MSTAR-like 切片（mstar_like/）
│   └── eval_gan_metrics.py         # 端到端：训练 R/E_asc → 计算六项指标 → 对照表
├── docs/
│   └── 11-生成模型评估指标.md        # 六项指标公式、实现要点、运行说明
├── requirements.txt
└── .gitignore
```

## 三、安装

```bash
pip install -r requirements.txt
# 纯 CPU 环境：pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

首次运行 `fid` 会自动下载 Inception-v3 的 IMAGENET1K_V1 预训练权重（约 104 MB），
之后缓存在本地。若离线失败会自动回退随机初始化（FID 仅作流程演示）。

## 四、复现步骤

### 方式 A：真实 MSTAR 数据（默认，推荐）

```bash
cd SAR-image-quality-assessment-v2
python examples/download_mstar.py      # ① 下载 .npz → mstar_raw/（约 256 MB，分片并行）
python examples/mstar_to_chips.py      # ② 转换 → mstar_real/（默认 10 类 × 72 角，128×128）
python examples/eval_gan_metrics.py    # ③ 训练 R/E_asc 并计算六项指标（--data 默认 mstar_real）
```

`download_mstar.py` 从 GitHub 仓库 [jwcalder/MSTAR-Active-Learning](https://github.com/jwcalder/MSTAR-Active-Learning)
下载 `Data/SAR10{a,b,c}.npz`（SDMS 官方公开 MSTAR mixed-targets 的预处理结果）。

`mstar_to_chips.py` 默认转换全部 10 类（2S1/BMP2/BRDM2/BTR60/BTR70/D7/T62/T72/ZIL131/ZSU23-4）
× 72 方位角（0–355°，5° 网格最近邻），17° 俯角，幅度切片 resize 到 128×128；
`--classes paper` 则只取论文 §3.1.1 的 5 类（2S1/BRDM2/D7/T62/ZIL131）。

### 方式 B：仿真 MSTAR-like 替身

```bash
python examples/generate_mstar_like.py   # 生成 mstar_like/
python examples/eval_gan_metrics.py --data mstar_like
```

## 五、预期输出（已在本机验证）

真实 MSTAR（`mstar_real`，10 类，300 epoch）：

```
================ 六项指标 ================
指标         real-real2(自洽)     real-fake(退化)    期望
FID                  0.01              6.97    自洽≈0 / 退化>0
SSIM               1.0000            0.9381    自洽高 / 退化更低
AFS                0.9999            0.9201    自洽≈1 / 退化<1
ΔENL               0.0023            0.2168    自洽≈0 / 退化>0
BVE                0.0001            0.0023    自洽≈0 / 退化>0
CMAE°               12.75             34.00    自洽小(估角残差) / 退化更差
```

两列对照设计意图：
- **real-real2（自洽）**：同一真实切片 vs 其「近乎相同的第二次观测」，应接近理想值 → 验证实现正确；
- **real-fake（退化）**：应明显偏离理想值 → 验证指标能区分生成质量。

CMAE：退化列（34.00°）明显高于自洽列（12.75°），说明角度控制误差被正确捕捉。
自洽列 12.75° 是 R 在 10 类目标上的估角残差（比 5 类时的 7.76° 更大——10 类角度估计更难）。
注意「退化列 − 自洽列」= 21.25° 并不严格等于 offset(10°)：退化施加的模糊+噪声也会系统性
偏移 R 的估角，5 类时该效应很小（≈ offset），10 类时 R 对退化更敏感而被放大。

## 六、实现口径（与论文的对齐点/偏差）

1. **数据源为幅度（magnitude）而非复数据**。论文用「原始 MSTAR 复数据」，但指标只用幅度；
   这里用 SDMS 公开幅度版（jwcalder 预处理，88×88 → resize 128×128）。
2. **ΔENL/BVE 在强度域（幅度²）计算**。ENL=μ²/σ² 的经典定义在强度/功率域。
3. **E_asc 训练协议论文未公开**。以浅层卷积编码器用重构自监督在真实切片上预训练作替身。
4. **FID 做 PCA 降维**。样本数远小于 Inception 的 2048 维，先联合 PCA 到 `min(64, N−1)` 维再算 Fréchet 距离。
5. **real2 自洽参照**。真实 MSTAR 每角度只有一次观测，无法像仿真那样换斑点种子；
   这里用 2% 乘性扰动模拟「近乎相同的第二次观测」，仅用于验证指标自洽性。
6. **R(·) 估角残差**。R 是论文口径的轻量 4 卷积块网络（16/32/64/128 → GAP → 2FC），
   在真实 10 类、5° 采样的 720 张上训练 300 epoch，估角残差约 13°（5 类子集约 7–8°）；CMAE 通过
   「退化列 − 自洽列 ≈ offset」体现角度控制误差，而非绝对归零。

> 数值是「实现正确性 + 指标行为」的验证，不与论文 Table 4 直接可比（论文训练目标、
> 生成器、R/E_asc 协议均未完全复现）。详见 [docs/11-生成模型评估指标.md](docs/11-生成模型评估指标.md)。
