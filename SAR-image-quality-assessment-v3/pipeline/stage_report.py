"""Stage E —— 报告：汇总各模型指标，产出对比 HTML / Markdown / JSON，并给出模型特征总结。

输出 results/：
  - summary.json   结构化指标汇总
  - report.md      Markdown 报告
  - report.html    科研风格 HTML（自包含，内联 CSS + 简单 SVG 柱状图）
"""
from __future__ import annotations

from common import io, paths
from evaluation.metrics_bridge import PAPER_KEYS, FR_KEYS

PAPER = PAPER_KEYS
FR = FR_KEYS

# 指标方向：+1 越大越好，-1 越小越好（报告侧独有，指标引擎不关心方向）
IDEAL = {
    "fid": -1, "ssim": +1, "afs": +1, "delta_enl": -1, "bve": -1, "cmae_deg": -1,
    "mse": -1, "rmse": -1, "psnr": +1, "ncc": +1, "uqi": +1, "ms_ssim": +1, "fsim": +1,
}


def _load_metrics(cfg):
    root = paths.res_dir(cfg, "metrics")
    out = {}
    if root.exists():
        for mdir in sorted(root.iterdir()):
            if mdir.is_dir():
                for f in sorted(mdir.glob("*.json")):
                    out[f"{mdir.name}/{f.stem}"] = io.read_json(f)
    return out


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.4f}"
    return str(v)


def _characterize_metrics(m):
    """绝对阈值 → (strength_labels, weak_labels, weak_metric_keys)。

    只覆盖跨任务稳定可比的指标（SSIM/AFS/ΔENL/NCC）。CMAE 与任务强相关（NVS/重构天然
    偏高），负向判定交给组内相对比较，这里只保留「准确」正向判定；负向的 CMAE 不再用绝对
    阈值硬判（否则会把组内最优的 adapted 误标成「偏差大」）。
    """
    strengths, weaknesses = [], []
    weak_metrics = set()

    if m.get("ssim") is not None:
        if m["ssim"] >= 0.6:
            strengths.append("结构保真度高（SSIM）")
        elif m["ssim"] < 0.35:
            weaknesses.append("结构保真度低（SSIM）")
            weak_metrics.add("ssim")
    if m.get("afs") is not None:
        if m["afs"] >= 0.9:
            strengths.append("目标区散射结构一致（AFS）")
        elif m["afs"] < 0.7:
            weaknesses.append("目标区散射结构偏差（AFS）")
            weak_metrics.add("afs")
    if m.get("delta_enl") is not None:
        if m["delta_enl"] < 0.5:
            strengths.append("背景统计一致性（ΔENL）")
        elif m["delta_enl"] > 2.0:
            weaknesses.append("背景噪声统计偏差（ΔENL）")
            weak_metrics.add("delta_enl")
    if m.get("cmae_deg") is not None and m["cmae_deg"] < 10:
        strengths.append("角度条件控制准确（CMAE）")
    if m.get("ncc") is not None:
        if m["ncc"] >= 0.5:
            strengths.append("与真实图相关性高（NCC）")
        elif m["ncc"] < 0.2:
            weaknesses.append("与真实图相关性低（NCC）")
            weak_metrics.add("ncc")

    return strengths, weaknesses, weak_metrics


def _characterize(name, m):
    strengths, weaknesses, _ = _characterize_metrics(m)
    return {"strengths": strengths, "weaknesses": weaknesses}


def _relative_weaknesses(rows):
    """按模型分组，标记「相对最弱」变体，但只挑差距显著且每变体至多 top-K 项（避免刷屏）。"""
    TOP_K = 3          # 每变体最多标记的相对弱项数
    MIN_GAP = 0.05     # 相对差距 <5% 视为噪声，不标
    groups = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    for model, rs in groups.items():
        if len(rs) < 2:
            continue
        deficits = []  # (相对差距, metric, worst_row)
        for metric in PAPER + FR:
            vals = [r["metrics"].get(metric) for r in rs]
            if any(v is None for v in vals):
                continue
            ideal = IDEAL[metric]
            best = min(vals) if ideal < 0 else max(vals)
            worst = max(vals) if ideal < 0 else min(vals)
            ref = max(abs(best), abs(worst), 1e-9)
            deficit = abs(worst - best) / ref
            if deficit < MIN_GAP:
                continue
            worst_row = max(rs, key=lambda r: r["metrics"][metric]) if ideal < 0 \
                else min(rs, key=lambda r: r["metrics"][metric])
            deficits.append((deficit, metric, worst_row))
        deficits.sort(key=lambda x: -x[0])
        marked = {}
        for _deficit, metric, row in deficits:
            if marked.get(id(row), 0) >= TOP_K:
                continue
            # 已被绝对阈值判为弱项的指标不再重复标「相对较弱」（避免同项出现两次）
            _, _, weak_metrics = _characterize_metrics(row["metrics"])
            if metric in weak_metrics:
                continue
            row["characterization"]["weaknesses"].append(f"相对较弱：{metric}")
            marked[id(row)] = marked.get(id(row), 0) + 1
    return rows


def _relative_strengths(rows):
    """与 _relative_weaknesses 对称：按模型分组，把每项指标的「组内相对最优」标到擅长列。

    - 只挑组内差距显著（>=MIN_GAP）且该项未被绝对阈值判为弱项的指标（避免同一项同时出现在
      擅长/不擅长两列，例如 baseline 的 SSIM 虽组内最优但仍很低，不写进「擅长」）。
    - 每变体至多 top-K 项（对称，防刷屏）。
    """
    TOP_K = 3
    MIN_GAP = 0.05
    groups = {}
    for r in rows:
        groups.setdefault(r["model"], []).append(r)
    for model, rs in groups.items():
        if len(rs) < 2:
            continue
        wins = []
        for metric in PAPER + FR:
            vals = [r["metrics"].get(metric) for r in rs]
            if any(v is None for v in vals):
                continue
            ideal = IDEAL[metric]
            best = min(vals) if ideal < 0 else max(vals)
            worst = max(vals) if ideal < 0 else min(vals)
            ref = max(abs(best), abs(worst), 1e-9)
            deficit = abs(worst - best) / ref
            if deficit < MIN_GAP:
                continue
            best_row = min(rs, key=lambda r: r["metrics"][metric]) if ideal < 0 \
                else max(rs, key=lambda r: r["metrics"][metric])
            _, _, weak_metrics = _characterize_metrics(best_row["metrics"])
            if metric in weak_metrics:
                continue
            wins.append((deficit, metric, best_row))
        wins.sort(key=lambda x: -x[0])
        marked = {}
        for _deficit, metric, row in wins:
            if marked.get(id(row), 0) >= TOP_K:
                continue
            row["characterization"]["strengths"].append(f"组内相对最优：{metric}")
            marked[id(row)] = marked.get(id(row), 0) + 1
    return rows


def _build_summary(cfg):
    metrics = _load_metrics(cfg)
    rows = []
    for key, m in metrics.items():
        model = m.get("model")
        rows.append({
            "key": key,
            "model": model, "variant": m.get("variant"),
            "n_pairs": m.get("n_pairs"), "note": m.get("note", ""),
            "pairing": cfg["models"].get(model, {}).get("pairing", ""),
            "metrics": {k: m.get(k) for k in PAPER + FR},
            "characterization": _characterize(key, m),
        })
    rows = _relative_weaknesses(rows)
    rows = _relative_strengths(rows)
    # 附受限项（数据受限但已按适用指标评估的模型）
    blocked = {}
    fp = paths.ws_dir(cfg, "generated", "frequency_gen", "generation_manifest.json")
    if fp.exists():
        man = io.read_json(fp)
        if man.get("data_limited"):
            blocked["frequency_gen"] = man
    return {"metrics": rows, "blocked": blocked,
            "class_names": cfg["datasets"]["class_names"]}


def _bar_svg(rows, key, title):
    """简单横向柱状图 SVG。"""
    vals = [(r["key"], r["metrics"].get(key)) for r in rows if r["metrics"].get(key) is not None]
    if not vals:
        return ""
    ideal = IDEAL.get(key, -1)
    vmax = max(abs(v) for _, v in vals) or 1.0
    h = 22
    w = 520
    bars = []
    y = 0
    for name, v in vals:
        # 越小越好则反转长度，统一用「离理想越远越长」不好展示；这里直接展示数值归一化
        frac = min(abs(v) / vmax, 1.0) * (w - 220)
        color = "#3b82f6"
        bars.append(
            f'<text x="0" y="{y + 15}" font-size="11" fill="#333">{name}</text>'
            f'<rect x="200" y="{y + 5}" width="{frac:.0f}" height="12" fill="{color}"/>'
            f'<text x="{200 + frac + 6:.0f}" y="{y + 15}" font-size="10" fill="#555">{_fmt(v)}</text>'
        )
        y += h
    return (f'<h4>{title}</h4><svg width="{w}" height="{y}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>')


def _html(cfg, summary):
    rows = summary["metrics"]
    # 概览表（六项 + 七项）
    head = "".join(f"<th>{k}</th>" for k in PAPER + FR)
    body = ""
    for r in rows:
        tds = "".join(f"<td>{_fmt(r['metrics'].get(k))}</td>" for k in PAPER + FR)
        body += (f"<tr><td class='k'>{r['key']}</td><td>{r.get('n_pairs','—')}</td>{tds}</tr>")

    blocked_html = ""
    for k, v in summary["blocked"].items():
        reason = v.get("reason") or v.get("note") or ""
        blocked_html += f"<tr><td>{k}</td><td>{reason}</td></tr>"

    chars = ""
    for r in rows:
        c = r["characterization"]
        s = "、".join(c["strengths"]) or "—"; w = "、".join(c["weaknesses"]) or "—"
        chars += (f"<tr><td class='k'>{r['key']}</td><td class='ok'>{s}</td>"
                  f"<td class='bad'>{w}</td></tr>")

    charts = ""
    for k in ["fid", "ssim", "cmae_deg", "delta_enl"]:
        charts += _bar_svg(rows, k, k)

    # FR 逐像素指标的配对口径（按模型去重）
    pairing = {}
    for r in rows:
        pairing.setdefault(r["model"], r.get("pairing", ""))
    pairing_html = "".join(f"<tr><td>{m}</td><td>{p or '—'}</td></tr>"
                           for m, p in pairing.items())

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>SAR 生成图像质检评估报告（v3）</title>
<style>
body{{font-family:-apple-system,'Segoe UI',Roboto,sans-serif;margin:32px auto;max-width:1080px;color:#222;line-height:1.5}}
h1{{border-bottom:2px solid #222;padding-bottom:8px}} h2{{margin-top:32px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:center}}
th{{background:#f4f4f4}} td.k{{text-align:left;font-weight:600}}
.ok{{color:#15803d}} .bad{{color:#b91c1c}} .note{{color:#666;font-size:12px}}
svg{{display:block;margin:8px 0}}
</style></head><body>
<h1>SAR 生成图像质检评估报告（v3）</h1>
<p class="note">评估端复用 v2（论文 §3.1.4 六项指标 + 七项 FR 全参考指标）；生成端覆盖 4 个子目录。</p>

<h2>1. 指标概览</h2>
<table><tr><th>模型/变体</th><th>配对数</th>{head}</tr>{body}</table>
<p class="note">理想方向：FID/ΔENL/BVE/CMAE/MSE/RMSE → 0；SSIM/AFS/NCC/UQI/MS-SSIM/FSIM → 1；PSNR → +∞。</p>

<h2>2. 关键指标对比</h2>{charts}

<h2>3. 模型特征总结（擅长 / 不擅长）</h2>
<table><tr><th>模型/变体</th><th>擅长</th><th>不擅长</th></tr>{chars}</table>

<h2>4. FR 配对口径（逐像素指标语义）</h2>
<p class="note">七项 FR 全参考指标依赖「真实↔生成」如何配对；不同模型配对方式不同，跨模型比较 FR 前需先对齐口径。</p>
<table><tr><th>模型</th><th>配对方式</th></tr>{pairing_html}</table>

<h2>5. 不可评估 / 受限项</h2>
<table><tr><th>模型</th><th>说明</th></tr>{blocked_html}</table>

<h2>6. 结论</h2>
<ul>
<li><b>StyleGAN4SAR</b>：10 类条件生成（128×128），基线 vs 增强（AASG+ESF+ENL）可直接对比插件增益。基线在多数分布/FR 指标上略优。</li>
<li><b>angle_gen</b>：NVS 视角生成（64×64），geometry（真几何姿态编码）vs baseline（针孔相机代理）姿态编码对比；adapted 为 geometry 同构的仓库默认 pretrain 权重（step 101715，低于 geometry 的 690000），用于单独分离「checkpoint/adaptation 选择」这一变量。注意：NVS 合成的是新视角而非重构，逐像素 FR 指标（SSIM/MSE/PSNR/NCC）天然偏低，应主要看 AFS/ΔENL 等物理/分布指标。</li>
<li><b>frequency_gen</b>：X→Ka Pix2Pix 翻译，无真实配对数据，仅 smoke 单样本；适用 SSIM + 7 项 FR 逐像素配对指标（FID/ΔENL/CMAE 等分布/角度指标不适用）。</li>
<li><b>GaussRecon4SAR</b>：3DGS 重构（T72，预生成 renders/gt，本机 Linux 无法重新生成）；按重构保真度评估（render vs gt 配对，全 13 项指标）。</li>
</ul>
</body></html>"""


def run(cfg, args):
    summary = _build_summary(cfg)
    res = paths.res_dir(cfg)
    res.mkdir(parents=True, exist_ok=True)
    io.write_json(res / "summary.json", summary)

    # Markdown
    md = ["# SAR 生成图像质检评估报告（v3）\n"]
    md.append("| 模型/变体 | 配对数 | " + " | ".join(PAPER + FR) + " |")
    md.append("|---|---|" + "---|" * len(PAPER + FR))
    for r in summary["metrics"]:
        md.append(f"| {r['key']} | {r.get('n_pairs','—')} | " +
                  " | ".join(_fmt(r['metrics'].get(k)) for k in PAPER + FR) + " |")
    md.append("\n## 模型特征\n")
    for r in summary["metrics"]:
        c = r["characterization"]
        md.append(f"- **{r['key']}**：擅长——{'、'.join(c['strengths']) or '—'}；不擅长——{'、'.join(c['weaknesses']) or '—'}")
    pairing = {}
    for r in summary["metrics"]:
        pairing.setdefault(r["model"], r.get("pairing", ""))
    md.append("\n## FR 配对口径（逐像素指标语义）\n")
    for m, p in pairing.items():
        md.append(f"- **{m}**：{p or '—'}")
    md.append("\n## 不可评估/受限\n")
    for k, v in summary["blocked"].items():
        md.append(f"- **{k}**：{v.get('reason') or v.get('note')}")
    (res / "report.md").write_text("\n".join(md), encoding="utf-8")

    (res / "report.html").write_text(_html(cfg, summary), encoding="utf-8")
    print(f"[stage_e] 报告已生成：{res}/report.html / report.md / summary.json "
          f"（{len(summary['metrics'])} 个评估条目）")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(paths.V3_ROOT))
    run(paths.load_config(), None)
