"""数据集级质检看板（自包含 HTML，复用单图报告的 CSS 与格式化工具）。

输入 `DatasetResult`（见 dataset.py），输出一份自包含 dashboard：
总体概览 → 评分分布 → 批次合格判定 → 聚合统计分布 → 离群检测 → 元数据一致性
→ 分组审计（留存率） → 每图明细（链接到 per_image/ 单图报告）。
"""
from __future__ import annotations

import html
from datetime import datetime

from .report import _CSS, _fmt, DIM_ORDER
from .dataset import _group_value

LEVEL_COLORS = {"优": "#2e7d32", "良": "#33691e", "合格": "#ef6c00", "不合格": "#c62828"}

_EXTRA_CSS = """
.bar-row{display:flex;align-items:center;gap:10px;margin:6px 0}
.bar-label{width:56px;color:var(--muted);font-size:12px;flex:none}
.bar{height:14px;border-radius:7px;background:#e5e7eb;overflow:hidden;flex:1}
.bar-fill{height:100%;border-radius:7px}
.table-scroll{overflow-x:auto}
h3{margin:18px 0 8px;font-size:15px}
"""


def _level_badge(score) -> str:
    if score is None:
        return "—"
    lv = "优" if score >= 90 else "良" if score >= 75 else "合格" if score >= 60 else "不合格"
    return f"<span style='color:{LEVEL_COLORS[lv]};font-weight:600'>{lv}</span>"


def _score_bars(sdist: dict) -> str:
    order = ["优", "良", "合格", "不合格", "error"]
    colors = {"优": "#2e7d32", "良": "#33691e", "合格": "#ef6c00",
              "不合格": "#c62828", "error": "#546e7a"}
    total = sdist.get("total", 0) or 1
    bars = []
    for k in order:
        c = sdist.get(k, 0)
        pct = c / total * 100.0
        bars.append(f"<div class='bar-row'><div class='bar-label'>{k}</div>"
                    f"<div class='bar'><div class='bar-fill' style='width:{pct:.1f}%;background:{colors[k]}'></div></div>"
                    f"<div class='mono' style='width:40px;text-align:right'>{c}</div></div>")
    return "".join(bars)


def _batch_section(result) -> str:
    b = result.batch
    samp = b.get("sampling")
    dc = b.get("defect_counts", {})
    samp_txt = html.escape(samp["reason"]) if samp else "未提供批量大小。"
    rows = (f"<table><tr><th style='width:180px'>图像数（成功）</th><td class='mono'>{b.get('n_images')}</td></tr>"
            f"<tr><th>整批结论</th><td><b>{html.escape(str(b.get('verdict')))}</b>"
            f"（等级 {html.escape(str(b.get('level')) or '—')}）</td></tr>"
            f"<tr><th>平均分 ± σ</th><td class='mono'>{_fmt(b.get('score_mean'))} ± {_fmt(b.get('score_std'))}"
            f"（范围 [{_fmt(b.get('score_min'))}, {_fmt(b.get('score_max'))}]）</td></tr>"
            f"<tr><th>缺陷汇总 A/B/C/D</th><td class='mono'>{dc.get('A', 0)}/{dc.get('B', 0)}/{dc.get('C', 0)}/{dc.get('D', 0)}</td></tr>"
            f"<tr><th>抽样建议（GB/T 24356）</th><td class='reason'>{samp_txt}</td></tr></table>")
    note = (f"<p style='color:#c62828;font-weight:600'>⚠ {html.escape(b['note'])}</p>"
            if b.get("note") else "")
    return f"<section><h2>批次合格判定（GB/T 24356）</h2>{note}{rows}</section>"


def _agg_row(a) -> str:
    oc = len(a.outliers)
    oc_txt = (f"<span style='color:#c62828;font-weight:600'>{oc}</span>" if oc else "0")
    std = f" ± {_fmt(a.std)}" if a.std is not None else ""
    return (f"<tr><td><b>{html.escape(a.name)}</b></td>"
            f"<td>{html.escape(a.unit)}</td>"
            f"<td class='mono'>{a.n_value}</td>"
            f"<td class='mono'>{a.n_pass}</td>"
            f"<td class='mono'>{a.n_warn}</td>"
            f"<td class='mono'>{a.n_fail}</td>"
            f"<td class='mono'>{a.n_nodata}</td>"
            f"<td class='mono'>{_fmt(a.mean)}{std}</td>"
            f"<td class='mono'>{_fmt(a.median)}</td>"
            f"<td class='mono'>{_fmt(a.min)}</td>"
            f"<td class='mono'>{_fmt(a.max)}</td>"
            f"<td>{oc_txt}</td></tr>")


def _aggregate_section(result) -> str:
    by_dim: dict[str, list] = {}
    for a in result.aggregates:
        by_dim.setdefault(a.dimension, []).append(a)
    sections = []
    for dim in DIM_ORDER:
        aggs = by_dim.get(dim)
        if not aggs:
            continue
        rows = "".join(_agg_row(a) for a in aggs)
        sections.append(
            f"<h3>{html.escape(dim)}（{len(aggs)} 项）</h3>"
            f"<div class='table-scroll'><table><thead><tr>"
            f"<th>指标</th><th>单位</th><th>样本</th><th>合格</th><th>关注</th><th>超差</th><th>无法评估</th>"
            f"<th>均值±σ</th><th>中位数</th><th>最小</th><th>最大</th><th>离群</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")
    return (f"<section><h2>聚合统计分布（{len(result.aggregates)} 项指标）</h2>"
            f"<p class='reason'>跨图汇总每项指标分布；「无法评估」不计入均值 / 分位统计。</p>"
            f"{''.join(sections)}</section>")


def _outliers_section(result) -> str:
    flagged: dict[str, list] = {}
    for a in result.aggregates:
        for o in a.outliers:
            flagged.setdefault(o["image"], []).append(
                {"key": a.key, "name": a.name, "value": o["value"], "z": o["z"], "kind": a.kind})
    if not flagged:
        return ("<section><h2>跨图一致性 / 离群检测</h2>"
                "<p class='reason'>未检出离群图像。</p></section>")
    items = []
    for image, flags in sorted(flagged.items(), key=lambda kv: -len(kv[1])):
        rows = "".join(
            f"<tr><td><b>{html.escape(f['name'])}</b></td><td class='mono'>{_fmt(f['value'])}</td>"
            f"<td class='mono'>{f['z']:.2f}</td>"
            f"<td>{'标记项（场景相关）' if f['kind'] == 'marker' else '判决项'}</td></tr>"
            for f in flags)
        items.append(
            f"<h3>{html.escape(image)}（{len(flags)} 项离群）</h3>"
            f"<table><thead><tr><th>指标</th><th>数值</th><th>|z|</th><th>类型</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")
    return ("<section><h2>跨图一致性 / 离群检测</h2>"
            "<p class='reason'>稳健离群（MAD / IQR）：以下图像在部分指标上显著偏离批内中位值，"
            "仅作人工复核提示，不自动判不合格。</p>" + "".join(items) + "</section>")


def _metadata_section(result) -> str:
    mc = result.metadata_consistency
    rows = []
    for f in mc["fields"]:
        vals = "、".join(f"{str(v['value'])}×{v['count']}" for v in f["values"])
        badge = "<span style='color:#2e7d32'>一致</span>" if f["consistent"] else \
                "<span style='color:#c62828'>不一致</span>"
        rows.append(f"<tr><td><b>{html.escape(f['key'])}</b></td>"
                    f"<td class='mono'>{f['n_present']}/{f['n_present'] + f['n_missing']}</td>"
                    f"<td>{html.escape(vals) or '—'}</td><td>{badge}</td></tr>")
    r = mc.get("rules", {})
    rules_txt = (f"元数据规则引擎（CEOS-ARD/CARD4L）：Threshold {r.get('threshold_achieved', 0)}/"
                 f"{r.get('threshold_total', 0)}，Goal {r.get('goal_achieved', 0)}/{r.get('goal_total', 0)}")
    return (f"<section><h2>元数据一致性</h2><p class='reason'>{html.escape(rules_txt)}</p>"
            f"<table><thead><tr><th>字段</th><th>覆盖</th><th>取值分布</th><th>一致性</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")


def _group_section(result) -> str:
    ga = result.group_audit
    if not ga.get("groups"):
        return "<section><h2>分组审计（留存率）</h2><p class='reason'>无可分组图像。</p></section>"
    rows = []
    for g in ga["groups"]:
        dc = g["defect_counts"]
        rows.append(f"<tr><td><b>{html.escape(g['group'])}</b></td><td class='mono'>{g['n']}</td>"
                    f"<td class='mono'>{g['mean_score']:.1f} ± {g['score_std']:.1f}</td>"
                    f"<td>{_level_badge(g['mean_score'])}</td>"
                    f"<td class='mono'>{dc.get('A', 0)}/{dc.get('B', 0)}/{dc.get('C', 0)}/{dc.get('D', 0)}</td>"
                    f"<td class='mono'>{g['n_fail']}</td>"
                    f"<td class='mono'>{g['retention_rate']}%</td></tr>")
    return (f"<section><h2>分组审计（留存率）</h2>"
            f"<p class='reason'>分组键：{html.escape(str(ga['group_by']))}；留存 = 非「不合格」图像占比。</p>"
            f"<table><thead><tr><th>分组</th><th>图像数</th><th>平均分±σ</th><th>等级</th>"
            f"<th>A/B/C/D</th><th>不合格数</th><th>留存率</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")


def _images_section(result) -> str:
    rows = []
    for r in result.records:
        if r.status == "error":
            st = "<span style='color:#c62828;font-weight:600'>失败</span>"
            score = "—"
        else:
            g = r.grade or {}
            score = _fmt(g.get("score"))
            lv = g.get("level")
            color = LEVEL_COLORS.get(lv, "#1a1a1a")
            st = f"<span style='color:{color};font-weight:600'>{html.escape(str(lv))}</span>"
        link = (f"<a href='{html.escape(r.report_html)}'>{html.escape(r.stem)}</a>"
                if r.report_html else html.escape(r.stem))
        counts = (r.grade or {}).get("counts") or {}
        err = (r.error or {}).get("message", "") if r.error else ""
        rows.append(f"<tr><td>{link}</td><td>{st}</td><td class='mono'>{score}</td>"
                    f"<td class='mono'>{counts.get('A', 0)}/{counts.get('B', 0)}/{counts.get('C', 0)}/{counts.get('D', 0)}</td>"
                    f"<td>{html.escape(_group_value(r, result.group_by))}</td>"
                    f"<td class='reason'>{html.escape(err)}</td></tr>")
    return (f"<section><h2>每图明细（{len(result.records)} 张）</h2>"
            f"<div class='table-scroll'><table><thead><tr>"
            f"<th>图像</th><th>等级</th><th>评分</th><th>A/B/C/D</th><th>分组</th><th>说明</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></section>")


def generate_dataset_html(result) -> str:
    """生成数据集级看板 HTML。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    b = result.batch
    n_total = len(result.records)
    n_ok = sum(1 for r in result.records if r.status == "ok")
    n_err = sum(1 for r in result.records if r.status == "error")
    dc = b.get("defect_counts", {})
    mean_txt = _fmt(b.get("score_mean"))
    verdict = b.get("verdict") or "无可用图像"
    vcolor = {"整批合格": "#2e7d32", "整批不合格": "#c62828"}.get(verdict, "#546e7a")

    hero = (f'<header class="hero"><h1>SAR 数据集级质检报告</h1>'
            f'<div class="sub">目录 {html.escape(result.root)} · 产品级 {html.escape(str(result.level))}'
            f' · 生成于 {now}</div></header>')

    tiles = f"""
    <div class="grid">
      <div class="tile score-tile"><div class="k">整批平均分</div>
        <div class="v" style="color:{vcolor}">{mean_txt}</div></div>
      <div class="tile"><div class="k">整批结论</div>
        <div class="v small" style="color:{vcolor}">{html.escape(verdict)}</div></div>
      <div class="tile"><div class="k">图像总数</div><div class="v small">{n_total}</div></div>
      <div class="tile"><div class="k">成功</div><div class="v small" style="color:#2e7d32">{n_ok}</div></div>
      <div class="tile"><div class="k">失败</div><div class="v small" style="color:#c62828">{n_err}</div></div>
    </div>
    <div class="legend">
      <span>缺陷汇总：A {dc.get('A', 0)} · B {dc.get('B', 0)} · C {dc.get('C', 0)} · D {dc.get('D', 0)}</span>
    </div>
    """

    bars_section = (f"<section><h2>评分分布</h2>{_score_bars(result.score_distribution)}</section>")

    body = (
        f"{hero}{tiles}{bars_section}"
        f"{_batch_section(result)}"
        f"{_aggregate_section(result)}"
        f"{_outliers_section(result)}"
        f"{_metadata_section(result)}"
        f"{_group_section(result)}"
        f"{_images_section(result)}"
        f'<div class="footer">SAR-IQA · 数据集级质检引擎 v0.1 · 批级判定依据 GB/T 24356，'
        f'最终验收需完整流水线（角反射器 / 时序 / 干涉 / 原始回波）</div>'
    )
    return (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>SAR 数据集级质检报告</title><style>{_CSS}{_EXTRA_CSS}</style>"
        f"</head><body><div class='wrap'>{body}</div></body></html>"
    )
