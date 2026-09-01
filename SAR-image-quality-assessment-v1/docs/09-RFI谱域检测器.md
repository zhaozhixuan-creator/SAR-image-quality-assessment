# 自研模块 · RFI 谱域检测器

> 对应调研空白 2。无成熟开源 Sentinel-1 RFI 检测 / 抑制工具（ESA / Aresys 业务化方案未开源）。**这是自动质检最成熟的切入点：问题定义清晰、特征在谱域显著、已有基准数据集、现成工具缺位。**

## 1. 需求背景

SAR 接收机带宽宽、增益高，与地面通信 / 雷达 / 导航系统共享频谱，RFI 在 C 波段尤其普遍。

**流行度**（Chojka 等 2020，Sensors 20(10):2919）：288 幅 Sentinel-1 SLC 中 113 幅（39%）受污染；中东 64/136 = 47%、东欧 46/99 = 46%、波兰仅 3/53 = 5.7%。**强地域依赖 → 全局阈值必然失效，必须逐场景检测。**

**检测成熟度**（Artiemjew 等 2021，Remote Sensing 13(1):7）：LeNet 类 CNN 中东强 RFI 准确率 92%，弱 RFI 仅 66%~84%。强干扰已可自动判定，"轻度污染但仍影响 InSAR 相干性"这一档最难识别，也正是质检最需要的一档。

## 2. 检测原理

RFI 应在**距离压缩前的谱域**做（图像域只能事后发现）。谱域是 RFI 最易识别的域：

- **窄带 RFI**：表现为距离频率—方位时间谱图上贯穿方位向的亮竖线。
- **宽带 RFI**：表现为局部方位时段的水平亮块。
- **图像域**：表现为方位向亮条纹，掩盖真实地物。

## 3. 算法设计

### 3.1 整体流程

```python
import numpy as np
from scipy.signal import find_peaks

def detect_rfi(slc_raw, noise_power=None):
    """
    slc_raw: 距离压缩前的原始回波（距离频率 x 方位时间）
    返回：RFI 掩膜 + 污染比例
    """
    # 1. 构建距离频率—方位时间谱
    spec = azimuth_fft(slc_raw)          # 方位向 FFT
    power = np.abs(spec) ** 2

    # 2. 行均值 3σ 准则（方位向，检测窄带竖线）
    row_mean = power.mean(axis=1)        # 距离频率维均值
    row_mask = (row_mean > row_mean.mean() + 3 * row_mean.std())

    # 3. 列均值 3σ 准则（距离频率维，检测宽带水平块）
    col_mean = power.mean(axis=0)        # 方位时间维均值
    col_mask = (col_mean > col_mean.mean() + 3 * col_mean.std())

    # 4. 合成污染掩膜
    rfi_mask = row_mask[:, None] | col_mask[None, :]

    # 5. 输出污染比例（作为可用性标记）
    ratio = rfi_mask.mean()
    return rfi_mask, ratio
```

### 3.2 窄带 vs 宽带判别

- **窄带**（连续波干扰，如 GNSS、地面通信载波）：在距离频率维上极窄、方位时间维上贯穿 → 用距离频率维均值 + 峰值检测。
- **宽带**（脉冲式干扰，如其他雷达）：在方位时间维上局部 → 用方位时间维均值 + 峰值检测。

### 3.3 输出口径

RFI 污染比例适合作为**场景可用性标记而非合格性判据**——同一颗卫星在不同地区成像，RFI 程度可相差极大，由下游应用决定是否接受。

## 4. 基准与验证

| 资源 | 说明 |
|---|---|
| SAR-IIDS（《雷达学报》） | Sentinel-1 干扰数据集，十余万幅筛选，6.81 GB，带 RFI 标注 |
| RFISD | RFI 标注数据集 |
| ESA / Aresys s1rfimap | 公开 Sentinel-1 RFI 地图服务（只发布结果不开放算法）——作为自研检测器的**外部交叉校验参考** |
| zephr-xyz/sentinel1-rfi-detection | S1 SLC 距离谱 GNSS 干扰检测，最完整但工程化有限，作算法参考 |

**免费数据源**：Sentinel-1 每个 burst 起始前 8~10 个回波（无发射、纯接收）可作为被动 C 波段频谱监测器（带宽 50~70 MHz），产品里本已存在供 RFI 质检使用的"免费"数据，只是通常未被利用。

## 5. 实现建议

1. 先做谱域阈值 + 3σ 的确定性检测（输出污染比例），再考虑 CNN 分类（强/弱/无三分类）。
2. 用 SAR-IIDS / RFISD 训练与评测，用 s1rfimap 做外部交叉校验。
3. 弱 RFI 检测是难点（影响 InSAR 相干性的一档），建议单独建模并给出置信度。
4. 输出为可用性元数据（污染比例 + 掩膜），随产品发布，不做合格性判决。
