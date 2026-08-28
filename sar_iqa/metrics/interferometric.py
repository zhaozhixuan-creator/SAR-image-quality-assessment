"""维度 6 · 干涉与相位质量（5 项）。

本维度所有指标只在多时相配对的相位域存在，单张图像（且为幅度/强度图）无法评估。
"""
from __future__ import annotations

from ..base import MetricResult, Status

DIM = "干涉与相位质量"


def compute(ctx) -> list[MetricResult]:
    reason = "需多时相复数据（干涉对 / 时序），单张图像无法评估干涉与相位质量。"
    return [
        MetricResult(key="interferometric.coherence", name="相干系数分布", dimension=DIM,
                     status=Status.NODATA, threshold="固定窗口相干估计（必须报告窗口）",
                     reason=reason),
        MetricResult(key="interferometric.phase_noise", name="相位噪声标准差", dimension=DIM,
                     status=Status.NODATA, threshold="高相干区相位离散度", reason=reason),
        MetricResult(key="interferometric.unwrap_residue", name="解缠残差点密度", dimension=DIM,
                     status=Status.NODATA, threshold="相位闭合残差", reason=reason),
        MetricResult(key="interferometric.orbit_ramp", name="轨道残余斜坡", dimension=DIM,
                     status=Status.NODATA, threshold="干涉图平面 / 二次曲面拟合系数", reason=reason),
        MetricResult(key="interferometric.aps", name="大气 / 电离层相位屏", dimension=DIM,
                     status=Status.NODATA, threshold="APS 估计与扣除后残差", reason=reason),
    ]
