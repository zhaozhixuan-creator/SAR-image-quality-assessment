# SAR-image-quality-assessment-v2

基于论文 *Remote Sensing* **remotesensing-18-02247**（SAR 未观测方位角图像生成，StyleGAN2 主干）
§3.1.4 "Evaluation Metrics" 的**生成模型评估指标**实现与验证。

> 与 [../SAR-image-quality-assessment-v1](../SAR-image-quality-assessment-v1)（单图质检引擎）相互独立：
> v2 评估的是「生成图像集」与「真实图像集」在六项指标上的**分布一致性**，用于给生成模型打分，
> 而非对单张 SAR 图像做 7 维 38 项质检。

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
│   ├── generate_mstar_like.py      # 生成仿真 MSTAR-like 切片
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
之后缓存在本地，无需重复下载。若离线失败会自动回退随机初始化（FID 仅作流程演示）。

## 四、复现步骤

```bash
cd SAR-image-quality-assessment-v2
python examples/generate_mstar_like.py     # ① 生成仿真切片（mstar_like/，约 504 张 128×128）
python examples/eval_gan_metrics.py        # ② 训练 R/E_asc 并计算六项指标
```

`generate_mstar_like.py` 关键参数：`--angle-offset 10.0`（生成集方位角偏移）、
`--looks-real 4.0` / `--looks-fake 3.2`（背景视数）、`--blur-sigma 0.4` / `--noise-frac 0.03`（目标退化）。

`eval_gan_metrics.py` 关键参数：`--angle-epochs 40`、`--asc-epochs 25`、`--fid-batch 8`、
`--no-fid`（跳过 FID，CPU 上 Inception 较慢时使用）。

## 五、预期输出（已在本机验证）

```
================ 六项指标 ================
指标         real-real2(自洽)     real-fake(退化)    期望
FID                  0.13              5.82    自洽≈0 / 退化>0
SSIM               0.9786            0.9560    自洽高 / 退化更低
AFS                1.0000            0.9845    自洽≈1 / 退化<1
ΔENL               0.0575            0.7891    自洽≈0 / 退化>0
BVE                0.0047            0.0620    自洽≈0 / 退化>0
CMAE°                2.07             10.44    自洽≈0(估角) / 退化≈offset(10)
```

两列对照的设计意图：
- **real-real2（自洽）**：同角度、不同斑点的两张干净图，应接近理想值 → 验证实现正确；
- **real-fake（退化）**：应明显偏离理想值 → 验证指标能区分生成质量。

数值回环校验：ΔENL 退化值 0.7891 ≈ |L_real−L_fake| = |4−3.2| = 0.8；
CMAE 退化值 10.44° ≈ 方位角偏移 offset=10°，说明 ΔENL 与 CMAE 的公式实现正确。

## 六、实现口径（与论文的对齐点/偏差）

1. **ΔENL/BVE 在强度域（幅度²）计算**。ENL=μ²/σ² 的经典定义在强度/功率域，L 视斑点
   只在强度域严格成立；在幅度域会得到错误方向。脚本里 `real_i = r**2` 即为此。
2. **E_asc 训练协议论文未公开**。论文仅说「预训练且冻结的 ASC 提取器」，这里以浅层卷积
   编码器用**重构自监督**在真实切片上预训练作替身，支撑 AFS 余弦相似度。
3. **FID 做 PCA 降维**。样本数（48）远小于 Inception 的 2048 维，直接算协方差不满秩；
   先联合 PCA 降到 `min(64, N−1)` 维再算 Fréchet 距离，是文档化的数值稳定措施。
4. **R(·) 与 E_asc 都很轻量**（128×128 输入、最多 128 通道），CPU 即可训练；
   Inception 是唯一较重的部分，但仅作 frozen 前向。
5. 掩码为论文的固定掩码：Mt = 中心 0.4H×0.4W 方形，Mb = 1−Mt，所有图共用。

> 数值是「实现正确性 + 指标行为」的验证；真实 MSTAR 数据不可得，切片为仿真替身，
> 因此不与论文 Table 4 直接可比。详见 [docs/11-生成模型评估指标.md](docs/11-生成模型评估指标.md)。
