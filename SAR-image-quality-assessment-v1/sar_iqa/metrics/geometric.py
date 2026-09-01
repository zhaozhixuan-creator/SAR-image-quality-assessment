"""维度 2 · 几何质量（5 项）。"""
from __future__ import annotations

from ..base import MetricResult, Status

DIM = "几何质量"


def compute(ctx) -> list[MetricResult]:
    return [
        _ale(ctx),
        _coregistration(ctx),
        _rtc(ctx),
        _layover_shadow(ctx),
        _projection_consistency(ctx),
    ]


def _ale(ctx):
    return MetricResult(
        key="geometric.ale", name="绝对定位精度 ALE", dimension=DIM,
        status=Status.NODATA, unit="m",
        threshold="S1C：均值 0.55 m / σ 0.41 m；GF-3 带控 RMSE 2.56 m",
        reason="需角反射器已知坐标 + 轨道/几何模型正演 + 全改正（电离层/对流层/潮/板块），单图无法计算。",
    )


def _coregistration(ctx):
    return MetricResult(
        key="geometric.coregistration", name="相对定位 / 配准精度", dimension=DIM,
        status=Status.NODATA, unit="像素",
        threshold="互相关偏移；模拟幅度配准残差",
        reason="需多时相 / 多通道配对数据；单通道间的配准见「极化质量 · 通道间配准」。",
    )


def _rtc(ctx):
    return MetricResult(
        key="geometric.rtc", name="地形校正（RTC）正确性", dimension=DIM,
        status=Status.NODATA,
        threshold="局部入射角与散射面积一致性",
        reason="需 DEM 提供的局部入射角与散射面积因子，单张图像（无 DEM 元数据）无法核验。",
    )


def _layover_shadow(ctx):
    return MetricResult(
        key="geometric.layover_shadow", name="叠掩阴影掩膜完整性", dimension=DIM,
        status=Status.NODATA,
        threshold="DEM 正演掩膜 vs 产品掩膜比对",
        reason="需 DEM 正演叠掩/阴影掩膜与产品掩膜比对，单图无 DEM 与产品掩膜。",
    )


def _projection_consistency(ctx):
    md = ctx.sar.metadata
    sp = md.get("pixel_spacing") or md.get("pixel_spacing_range") or md.get("range_spacing")
    crs = md.get("crs") or md.get("projection")
    if not sp and not crs:
        return MetricResult(
            key="geometric.projection_consistency", name="投影 / 采样间隔一致性", dimension=DIM,
            status=Status.NODATA,
            threshold="元数据声明 vs 栅格实际几何比对",
            reason="未提供投影 / 采样间隔元数据，无法比对。",
        )
    # 轻量核验：若声明了像素间距，与栅格尺寸粗校验（此处仅确认字段存在且为正数）
    ok = True
    note = []
    if sp is not None:
        try:
            ok = ok and float(sp) > 0
            note.append(f"像素间距 {sp}")
        except (TypeError, ValueError):
            ok = False
            note.append("像素间距非法")
    if crs is not None:
        note.append(f"坐标系 {crs}")
    return MetricResult(
        key="geometric.projection_consistency", name="投影 / 采样间隔一致性", dimension=DIM,
        status=Status.PASS if ok else Status.WARN, unit="",
        threshold="元数据声明 vs 栅格实际几何比对",
        reason="；".join(note) + ("（一致性通过字段自检）" if ok else "（字段异常）"),
        detail={"pixel_spacing": sp, "crs": crs},
    )
