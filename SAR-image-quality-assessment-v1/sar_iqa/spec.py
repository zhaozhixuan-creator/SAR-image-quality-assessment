"""规格单一事实源：7 维 38 项质检指标的结构化定义。

把《SAR 图像质检实现方案》(README) 中分散在各节的元信息收敛为一份可查询数据，
供分级流水线（pipeline）、判据与缺陷分级（grading）、开源生态选型（ecosystem）、
报告（report）共同驱动，避免在各处重复硬编码。

每一项指标的字段严格取自 README：
- level   : 原生产品级别（§二 分级质检流水线）——「指标在其原生产品级别测量」。
- phase   : 分阶段落地路径（§四 三期）——按成本递增原则。
- kind    : 判决项 vs 标记项（§五.4）——标记项只作可用性元数据、不计分。
- method  : 测量方法（§三 指标表）。
- criteria: 判据 / 基准（§三 指标表）。
- refs    : 参考开源实现（§三「实现」列 + §八 参考开源生态），键见 ecosystem.TOOLS。

级别常量：
- L0     原始级（下行完整性、丢包）
- L1/SLC 单视复级（物理性能指标的唯一正确测量位置）
- L2     地距 / 地理编码级（几何与地形）
- L3+    分析就绪级（ARD 合规与相位质量）
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 产品级别
L0 = "L0"
L1_SLC = "L1/SLC"
L2 = "L2"
L3 = "L3+"
UNIFIED = "统一判定"  # 缺陷分级 / 批抽样判定，非指标所在级

LEVELS = [L0, L1_SLC, L2, L3]

# 判决项 / 标记项
DECISION = "decision"
MARKER = "marker"

# 三期落地路径
PHASE_1 = 1  # 低成本全量前置筛查（一维 / 直方图 / 连通域统计）
PHASE_2 = 2  # sct 引擎核心物理指标（SLC 级）
PHASE_3 = 3  # 自研补缺模块 + 元数据合规 + 干涉 / 时序


@dataclass(frozen=True)
class MetricSpec:
    """单个指标的规格元信息（不含数值，数值由 metrics 各维度计算）。"""

    key: str                 # 稳定英文标识，与 MetricResult.key 一致
    name: str                # 中文名
    dimension: str           # 所属维度
    level: str               # 原生产品级别
    phase: int               # 落地分期 1/2/3
    kind: str                # decision（判决项）/ marker（标记项）
    method: str              # 测量方法
    criteria: str = ""       # 判据 / 基准
    refs: tuple[str, ...] = ()  # 参考开源实现（工具键，见 ecosystem.TOOLS）


# ---------------------------------------------------------------------------
# 38 项指标定义。维度顺序与 README §三 一致。
# ---------------------------------------------------------------------------
SPEC: tuple[MetricSpec, ...] = (
    # ---------- 维度 1 · 辐射质量（6 项） ----------
    MetricSpec("radiation.absolute_calibration", "绝对辐射定标精度", "辐射质量", L1_SLC, PHASE_2, DECISION,
               "角反射器 RCS 反演 vs 理论 RCS",
               "GF-3：1.3~1.4 dB；S1C 实测 0.36 dB(1σ)；工程门槛 1 dB(3σ)",
               ("sct", "gecoris")),
    MetricSpec("radiation.stability", "辐射稳定性 / 长期漂移", "辐射质量", L3, PHASE_3, DECISION,
               "雨林等稳定分布目标 γ⁰ 时序",
               "LT-1A：<1 dB/1000 km、<0.5 dB/5 天",
               ("self", "version-ledger")),
    MetricSpec("radiation.eap_residual", "跨轨天线方向图（EAP）残差", "辐射质量", L1_SLC, PHASE_1, DECISION,
               "均匀场景列均值沿幅宽剖面",
               "S1C 俯仰向总增益变化 0.3 dB(1σ)；波位间偏移 ≤ 0.2 dB",
               ("self",)),
    MetricSpec("radiation.subband_consistency", "子带 / 拼接辐射一致性", "辐射质量", L1_SLC, PHASE_1, DECISION,
               "子带交界列均值阶跃",
               "交界处无可见台阶",
               ("self", "sentinel1denoised")),
    MetricSpec("radiation.scalloping", "扇贝效应残差", "辐射质量", L1_SLC, PHASE_1, DECISION,
               "方位向行均值周期性起伏",
               "剖面峰峰值 / 标准差(dB)",
               ("sct",)),
    MetricSpec("radiation.saturation", "饱和率 / 动态范围", "辐射质量", L1_SLC, PHASE_1, DECISION,
               "直方图上界堆积计数",
               "饱和率 → 缺陷类（B 类 / 重）",
               ("self",)),

    # ---------- 维度 2 · 几何质量（5 项） ----------
    MetricSpec("geometric.ale", "绝对定位精度 ALE", "几何质量", L2, PHASE_2, DECISION,
               "角反射器预测（轨道 + 几何正演）vs 实测亚像素峰值 + 全改正",
               "S1C：均值 0.55 m / σ 0.41 m；GF-3 带控 RMSE 2.56 m、无控 < 50 m",
               ("sct",)),
    MetricSpec("geometric.coregistration", "相对定位 / 配准精度", "几何质量", L2, PHASE_2, DECISION,
               "互相关偏移；模拟幅度配准残差", "",
               ("isce2", "isce3", "mintpy")),
    MetricSpec("geometric.rtc", "地形校正（RTC）正确性", "几何质量", L2, PHASE_2, DECISION,
               "局部入射角与散射面积一致性", "",
               ("sarsen", "isce3")),
    MetricSpec("geometric.layover_shadow", "叠掩阴影掩膜完整性", "几何质量", L2, PHASE_2, MARKER,
               "DEM 正演掩膜 vs 产品掩膜比对", "",
               ("sarsen", "s1ard", "opensartoolkit")),
    MetricSpec("geometric.projection_consistency", "投影 / 采样间隔一致性", "几何质量", L2, PHASE_2, DECISION,
               "元数据声明 vs 栅格实际几何比对", "",
               ("stac-check", "self")),

    # ---------- 维度 3 · 分辨率与点目标响应（5 项） ----------
    MetricSpec("resolution.resolution", "空间分辨率（−3 dB）", "分辨率与点目标", L1_SLC, PHASE_2, DECISION,
               "16× 过采样 IRF 主瓣半功率点宽度",
               "矩形谱理论值 0.886 分辨单元；标称 / 实测 / 像素间距三者不可混用",
               ("sct",)),
    MetricSpec("resolution.pslr", "PSLR 峰值旁瓣比", "分辨率与点目标", L1_SLC, PHASE_2, DECISION,
               "最高旁瓣峰值 / 主瓣峰值",
               "加权后 < −20 dB；GF-3 < −22 dB；未加权 sinc 基线 −13.3 dB（引用陷阱）",
               ("sct",)),
    MetricSpec("resolution.islr", "ISLR 积分旁瓣比", "分辨率与点目标", L1_SLC, PHASE_2, DECISION,
               "全部旁瓣能量 / 主瓣能量（积分）",
               "< −15 dB；GF-3 < −15 dB；未加权 sinc 基线 −9.97 dB",
               ("sct",)),
    MetricSpec("resolution.sslr", "SSLR 次生旁瓣比", "分辨率与点目标", L1_SLC, PHASE_2, DECISION,
               "非邻近旁瓣（成对回波 / 幽灵）/ 主瓣", "",
               ("sct",)),
    MetricSpec("resolution.broadening", "散焦 / 展宽因子", "分辨率与点目标", L1_SLC, PHASE_2, DECISION,
               "实测主瓣宽度 / 理论主瓣宽度",
               "≈ 1 聚焦正常，显著 > 1 为聚焦退化",
               ("sct", "self")),

    # ---------- 维度 4 · 噪声与干扰（6 项） ----------
    MetricSpec("noise.nesz", "NESZ 噪声等效散射系数", "噪声与干扰", L1_SLC, PHASE_2, DECISION,
               "极低后向散射区（Doldrums）交叉极化剖面",
               "TSX −19~−26 dB；GF-3 设计 −20 dB / 在轨约 −22 dB",
               ("sct", "xarray-sentinel")),
    MetricSpec("noise.enl", "ENL / 辐射分辨率", "噪声与干扰", L1_SLC, PHASE_3, DECISION,
               "均匀区矩估计 + 对数累积量估计（双估计量）",
               "GF-3 辐射分辨率 3 dB；ENL 须与 EPD-ROA 成对报告",
               ("self",)),
    MetricSpec("noise.ambiguity", "AASR / RASR 模糊比", "噪声与干扰", L1_SLC, PHASE_2, DECISION,
               "方位谱复制能量占比（AASR）；鬼影识别（RASR）",
               "TSX 方位 ≈ −20 dB、距离 ≈ −25 dB",
               ("sct",)),
    MetricSpec("noise.rfi", "RFI 污染比例", "噪声与干扰", L1_SLC, PHASE_3, MARKER,
               "距离频率—方位时间谱异常 + 行均值 3σ",
               "场景可用性标记，非合格性判据（强地域依赖，须逐场景检测）",
               ("self", "sentinel1-rfi-detection")),
    MetricSpec("noise.negative_rate", "去噪负值率", "噪声与干扰", L1_SLC, PHASE_1, DECISION,
               "热噪声减除后负像素占比",
               "过高 = 噪声矢量被高估；过低 = 噪声未被充分减除",
               ("self", "sentinel1denoised")),
    MetricSpec("noise.dropout", "丢行 / 坏像元率", "噪声与干扰", L0, PHASE_1, DECISION,
               "行/列统计突变检测 + 无效值连通域分析",
               "A（严重，大面积）/ D（轻，零星）",
               ("self",)),

    # ---------- 维度 5 · 极化质量（5 项） ----------
    MetricSpec("polarimetric.amplitude_imbalance", "通道幅度不平衡", "极化质量", L1_SLC, PHASE_2, DECISION,
               "VV/HH 沿距离向比值",
               "S1C −0.5~+0.7 dB；LT-1A ≈ 0.2 dB",
               ("sct", "self")),
    MetricSpec("polarimetric.phase_imbalance", "通道相位不平衡", "极化质量", L1_SLC, PHASE_2, DECISION,
               "同极化相位差统计", "LT-1A ≈ 2°",
               ("sct", "self")),
    MetricSpec("polarimetric.crosstalk", "交叉极化串扰", "极化质量", L1_SLC, PHASE_2, DECISION,
               "角反射器或分布目标反演串扰矩阵",
               "LT-1A < −35 dB；ALOS-2 实测 −38 dB",
               ("sct",)),
    MetricSpec("polarimetric.reciprocity", "互易性 VH≈HV", "极化质量", L1_SLC, PHASE_1, DECISION,
               "两通道直方图与逐像素比对", "单站 SAR 应统计一致",
               ("self",)),
    MetricSpec("polarimetric.coregistration", "通道间配准", "极化质量", L1_SLC, PHASE_1, DECISION,
               "通道间互相关偏移", "各极化通道空间对齐",
               ("self",)),

    # ---------- 维度 6 · 干涉与相位质量（5 项） ----------
    MetricSpec("interferometric.coherence", "相干系数分布", "干涉与相位质量", L3, PHASE_3, MARKER,
               "固定窗口相干估计（必须报告窗口）", "只对处理退相干设阈值",
               ("sct", "isce2", "isce3", "dolphin")),
    MetricSpec("interferometric.phase_noise", "相位噪声标准差", "干涉与相位质量", L3, PHASE_3, DECISION,
               "高相干区相位离散度", "",
               ("mintpy", "gecoris")),
    MetricSpec("interferometric.unwrap_residue", "解缠残差点密度", "干涉与相位质量", L3, PHASE_3, DECISION,
               "相位闭合（loop closure）残差", "",
               ("licsbas", "mintpy")),
    MetricSpec("interferometric.orbit_ramp", "轨道残余斜坡", "干涉与相位质量", L3, PHASE_3, DECISION,
               "干涉图平面 / 二次曲面拟合系数", "",
               ("mintpy", "gmtsar")),
    MetricSpec("interferometric.aps", "大气 / 电离层相位屏", "干涉与相位质量", L3, PHASE_3, DECISION,
               "APS 估计与扣除后残差", "",
               ("raider", "pyaps", "pysolid")),

    # ---------- 维度 7 · 完整性与元数据（6 项） ----------
    MetricSpec("integrity.coverage", "有效数据覆盖率", "完整性与元数据", L3, PHASE_1, DECISION,
               "有效像元 / 标称幅宽", "",
               ("s1tiling", "self")),
    MetricSpec("integrity.fill_consistency", "无效值 / 填充标识", "完整性与元数据", L3, PHASE_1, DECISION,
               "掩膜 vs 实际无效区一致性", "B（重，未标记）/ D（轻，已标记）",
               ("self",)),
    MetricSpec("integrity.field_completeness", "元数据字段齐备性", "完整性与元数据", L3, PHASE_3, DECISION,
               "对照 CEOS-ARD Threshold / Goal 条款逐条自评", "",
               ("stac-check", "s1ard")),
    MetricSpec("integrity.calibration_traceability", "定标常数可追溯", "完整性与元数据", L3, PHASE_3, DECISION,
               "定标因子、辅助数据 / 处理器版本审计", "",
               ("self", "version-ledger")),
    MetricSpec("integrity.mask_layer", "掩膜与不确定度层", "完整性与元数据", L3, PHASE_3, DECISION,
               "是否提供逐像素质量层", "",
               ("s1ard",)),
    MetricSpec("integrity.card4l", "CARD4L 合规等级", "完整性与元数据", L3, PHASE_3, DECISION,
               "逐条自评 Threshold / Goal", "",
               ("s1ard", "stac-check")),
)

SPEC_BY_KEY: dict[str, MetricSpec] = {s.key: s for s in SPEC}


def metric_spec(key: str) -> MetricSpec | None:
    """按 key 取规格；未知 key 返回 None（如内部错误占位项）。"""
    return SPEC_BY_KEY.get(key)


def metrics_for_level(level: str) -> list[MetricSpec]:
    """原生级为 level 的指标。"""
    return [s for s in SPEC if s.level == level]


def metrics_for_phase(phase: int) -> list[MetricSpec]:
    """属于某期落地路径的指标。"""
    return [s for s in SPEC if s.phase == phase]


def marker_keys() -> set[str]:
    """标记项 key 集合（场景相关退化，只作可用性元数据、不计分）。"""
    return {s.key for s in SPEC if s.kind == MARKER}


def level_counts() -> dict[str, int]:
    """各级指标数（L0/L1-SLC/L2/L3+）。"""
    return {lv: sum(1 for s in SPEC if s.level == lv) for lv in LEVELS}


def phase_counts() -> dict[int, int]:
    """各期指标数。"""
    return {p: sum(1 for s in SPEC if s.phase == p) for p in (PHASE_1, PHASE_2, PHASE_3)}


def annotate(results) -> None:
    """按 key 把 spec 元信息回填到 MetricResult（level/phase/kind/method/refs）。

    幂等：未知 key 保持字段原样；已有值不覆盖（默认值除外）。
    """
    for r in results:
        s = SPEC_BY_KEY.get(r.key)
        if s is None:
            continue
        r.level = s.level
        r.phase = s.phase
        r.kind = s.kind
        r.method = s.method
        r.refs = list(s.refs)
