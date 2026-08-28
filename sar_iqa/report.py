"""HTML 质检报告生成（自包含，图表以 base64 内嵌）。"""
from __future__ import annotations

import html
from datetime import datetime

from .base import MetricResult, STATUS_META, Status
from . import plots

DIM_ORDER = ["辐射质量", "几何质量", "分辨率与点目标", "噪声与干扰",
             "极化质量", "干涉与相位质量", "完整性与元数据"]

_CSS = """
:root{--ink:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--bg:#fafafa;--card:#fff}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,'Segoe UI','Microsoft YaHei',Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink);background:var(--bg);line-height:1.55;font-size:14px}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
header.hero{background:linear-gradient(135deg,#0f2b46,#14547f);color:#fff;border-radius:14px;padding:26px 30px}
header.hero h1{margin:0 0 4px;font-size:22px}
header.hero .sub{opacity:.85;font-size:13px}
.grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;flex:1 1 150px}
.tile .k{color:var(--muted);font-size:12px}
.tile .v{font-size:26px;font-weight:700;margin-top:2px}
.tile .v.small{font-size:18px}
.score-tile .v{font-size:40px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-top:20px}
h2{font-size:17px;margin:0 0 14px;border-left:4px solid #14547f;padding-left:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;background:#f5f7fa}
tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.mono{font-variant-numeric:tabular-nums}
img.chart{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:6px 4px 6px 0}
.charts{display:flex;flex-wrap:wrap;gap:8px}
.reason{color:var(--muted);font-size:12px}
.nodata-list li{margin:4px 0;font-size:13px}
.footer{margin-top:24px;color:var(--muted);font-size:12px;text-align:center}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
"""


def _badge(status: Status) -> str:
    m = STATUS_META[status]
    return (f'<span class="badge" style="color:{m["color"]};background:{m["bg"]}">'
            f'{m["label"]}</span>')


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        av = abs(v)
        if av == 0:
            return "0"
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        if av >= 1:
            return f"{v:.2f}"
        return f"{v:.3f}"
    return str(v)


def _metric_rows(results: list[MetricResult]) -> str:
    rows = []
    for r in results:
        val = _fmt(r.value)
        rows.append(
            f"<tr><td><b>{html.escape(r.name)}</b></td>"
            f"<td class='mono'>{val}</td>"
            f"<td>{html.escape(r.unit) if r.unit else ''}</td>"
            f"<td>{_badge(r.status)}</td>"
            f"<td class='reason'>{html.escape(r.threshold or '')}</td>"
            f"<td class='reason'>{html.escape(r.reason or '')}</td></tr>"
        )
    return "\n".join(rows)


def generate_html(results: list[MetricResult], ctx, grade: dict,
                  despeckling: dict | None = None,
                  rule_results: list | None = None) -> str:
    sar = ctx.sar

    # 概览统计
    from collections import Counter
    counts = Counter(r.status for r in results)
    total = len(results)

    # 图表
    thumb = plots.thumbnail(sar)
    prof_img = plots.profiles(ctx.row_prof, ctx.col_prof)
    hist_img = plots.histogram(sar.intensity)
    irf_img = plots.irf(ctx.irf) if ctx.irf else ""

    # 概览卡片
    level_color = {"优": "#2e7d32", "良": "#33691e", "合格": "#ef6c00", "不合格": "#c62828"}[grade["level"]]
    cards = f"""
    <div class="grid">
      <div class="tile score-tile"><div class="k">质检评分</div>
        <div class="v" style="color:{level_color}">{grade['score']:.1f}</div></div>
      <div class="tile"><div class="k">质检等级</div>
        <div class="v" style="color:{level_color}">{grade['level']}</div></div>
      <div class="tile"><div class="k">指标总数</div><div class="v small">{total}</div></div>
      <div class="tile"><div class="k">合格</div><div class="v small" style="color:#2e7d32">{counts.get(Status.PASS,0)}</div></div>
      <div class="tile"><div class="k">需关注</div><div class="v small" style="color:#ef6c00">{counts.get(Status.WARN,0)}</div></div>
      <div class="tile"><div class="k">超差</div><div class="v small" style="color:#c62828">{counts.get(Status.FAIL,0)}</div></div>
      <div class="tile"><div class="k">无法评估</div><div class="v small" style="color:#546e7a">{counts.get(Status.NODATA,0)}</div></div>
    </div>
    <div class="legend">
      <span>缺陷：A {grade['counts']['A']} · B {grade['counts']['B']} · C {grade['counts']['C']} · D {grade['counts']['D']}</span>
      <span>已评估 {grade['evaluated']} 项 / 共 {total} 项</span>
    </div>
    """

    if grade.get("note"):
        cards += f'<p style="color:#c62828;font-weight:600;margin-top:12px">⚠ {html.escape(grade["note"])}</p>'

    # 输入信息
    md_keys = ", ".join(sar.metadata.keys()) if sar.metadata else "无"
    input_info = f"""
    <section><h2>输入信息</h2>
    <table>
      <tr><th style="width:160px">文件</th><td>{html.escape(sar.path)}</td></tr>
      <tr><th>尺寸</th><td>{sar.W} × {sar.H}（宽 × 高）</td></tr>
      <tr><th>通道</th><td>{sar.n_channels}（{'、'.join(sar.channel_names)}）</td></tr>
      <tr><th>输入域</th><td>{html.escape(sar.domain)}</td></tr>
      <tr><th>元数据边车</th><td>{html.escape(md_keys)}</td></tr>
    </table>
    <div class="charts">{_img(thumb)}{_img(prof_img)}{_img(hist_img)}{_img(irf_img)}</div>
    </section>
    """

    # 各维度明细
    dim_sections = []
    for dim in DIM_ORDER:
        dim_results = [r for r in results if r.dimension == dim]
        if not dim_results:
            continue
        dim_sections.append(
            f"<section><h2>{html.escape(dim)}（{len(dim_results)} 项）</h2>"
            f"<table><thead><tr><th style='width:180px'>指标</th><th>数值</th><th>单位</th>"
            f"<th>判定</th><th>判据 / 基准</th><th>说明</th></tr></thead>"
            f"<tbody>{_metric_rows(dim_results)}</tbody></table></section>"
        )

    # 自研模块结果
    module_section = _despeckling_section(despeckling) if despeckling else ""
    if rule_results is not None:
        module_section += _rules_section(rule_results)

    # 缺陷清单
    defect_section = _defect_section(grade)

    # 需外部数据
    nodata = [r for r in results if r.status == Status.NODATA]
    nodata_section = ""
    if nodata:
        items = "".join(
            f"<li><b>{html.escape(r.name)}</b>（{html.escape(r.dimension)}）：{html.escape(r.reason)}</li>"
            for r in nodata
        )
        nodata_section = f"<section><h2>需外部数据项（{len(nodata)} 项）</h2><ul class='nodata-list'>{items}</ul></section>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f'<header class="hero"><h1>SAR 图像质检报告</h1>'
        f'<div class="sub">7 维 38 项指标体系 · 单图质检引擎 · 生成于 {now}</div></header>'
        f"{cards}{input_info}{''.join(dim_sections)}{module_section}{defect_section}{nodata_section}"
        f'<div class="footer">SAR-IQA · 单图质检引擎 v0.1 · 结果基于单张图像，最终验收需完整流水线（角反射器/时序/干涉/原始回波）</div>'
    )
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>SAR 图像质检报告</title><style>{_CSS}</style></head><body><div class='wrap'>{body}</div></body></html>"
    )


def _img(b64: str) -> str:
    return f'<img class="chart" src="{b64}">' if b64 else ""


def _despeckling_section(d: dict) -> str:
    enl = d.get("enl", {})
    ratio = d.get("ratio", {})
    epd = d.get("epd", {})
    m = d.get("m_index")
    rows = f"""
    <tr><td style='width:200px'><b>ENL 矩估计</b></td><td class='mono'>{_fmt(enl.get('enl_moment'))}</td><td>视数</td>
      <td class='reason'>对数累积量估计 {_fmt(enl.get('enl_logcumulant'))}（分歧={enl.get('diverged', False)}）</td></tr>
    <tr><td><b>比值图均值偏差</b></td><td class='mono'>{_fmt(ratio.get('mean_bias'))}</td><td>—</td>
      <td class='reason'>应 ≈ 0（比值图应逼近纯斑点）</td></tr>
    <tr><td><b>比值图方差偏差</b></td><td class='mono'>{_fmt(ratio.get('var_bias'))}</td><td>—</td>
      <td class='reason'>单视指数分布理论方差 = 1</td></tr>
    <tr><td><b>比值图空间自相关</b></td><td class='mono'>{_fmt(ratio.get('spatial_ac'))}</td><td>—</td>
      <td class='reason'>≈ 0 为白噪声；≈ 1 表示残留地物结构（过平滑信号）</td></tr>
    <tr><td><b>EPD-ROA 边缘保持度</b></td><td class='mono'>{_fmt(epd.get('epd'))}</td><td>—</td>
      <td class='reason'>≈ 1 表示边缘对比度完整保留（{epd.get('n_edges', 0)} 个边缘点）</td></tr>
    <tr><td><b>M-index（近似）</b></td><td class='mono'>{_fmt(m)}</td><td>—</td>
      <td class='reason'>越小越好；一阶偏差 + 残留结构合成</td></tr>
    """
    return (f"<section><h2>自研模块 · 去斑评价指标包</h2>"
            f"<p class='reason'>参考去斑为内部简化 Lee 滤波基线；ENL 与 EPD-ROA 须成对解读，"
            f"单看 ENL 会被过度平滑刷分。</p>"
            f"<table><thead><tr><th style='width:200px'>指标</th><th>数值</th><th>单位</th><th>说明</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>")


def _rules_section(results: list) -> str:
    from .base import Status
    rows = []
    for r in results:
        if r.passed is None:
            badge = _badge(Status.NODATA)
            st = "N/A"
        elif r.passed:
            badge = _badge(Status.PASS)
            st = r.achieved
        else:
            badge = _badge(Status.WARN)
            st = "未达成"
        rows.append(
            f"<tr><td><b>{html.escape(r.name)}</b></td><td>{html.escape(r.clause)}</td>"
            f"<td>{html.escape(r.level)}</td><td>{html.escape(st)}</td><td>{badge}</td>"
            f"<td class='reason'>{html.escape(r.reason)}</td></tr>"
        )
    return (f"<section><h2>自研模块 · 元数据规则引擎（CEOS-ARD / CARD4L）</h2>"
            f"<table><thead><tr><th>规则</th><th>条款</th><th>级别</th><th>达成</th><th>判定</th><th>说明</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")


def _defect_section(grade: dict) -> str:
    defects = grade["defects"]
    if not defects:
        return "<section><h2>缺陷清单</h2><p class='reason'>未检出缺陷。</p></section>"
    rows = []
    for d in defects:
        color = {"A": "#c62828", "B": "#d84315", "C": "#ef6c00", "D": "#9e9d24"}[d["class"]]
        rows.append(
            f"<tr><td><b>{html.escape(d['name'])}</b></td><td>{html.escape(d['dimension'])}</td>"
            f"<td><span class='badge' style='color:#fff;background:{color}'>{d['class']} 类</span></td>"
            f"<td class='mono'>{_fmt(d['value'])} {html.escape(d['unit'] or '')}</td>"
            f"<td class='reason'>{html.escape(d['reason'])}</td></tr>"
        )
    return (f"<section><h2>缺陷清单（超差比例 → A/B/C/D）</h2>"
            f"<table><thead><tr><th>指标</th><th>维度</th><th>缺陷类</th><th>数值</th><th>说明</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")
