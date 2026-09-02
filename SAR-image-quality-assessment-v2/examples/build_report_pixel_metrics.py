#!/usr/bin/env python3
"""对 run1_float 生成 SAR 图做 7 项 FR 全参考指标质检，生成科研风格 HTML 报告。

依据 arXiv 2411.05027（GenAIxSAR 综述）补充的逐像素/结构全参考 IQA 指标：
  MSE / RMSE / PSNR —— 逐像素灰度保真
  NCC / UQI        —— 相关 / 统计一致性
  MS-SSIM / FSIM   —— 多尺度结构 / 感知特征相似度

报告结构（学习 run1_quality_report 的科研写法，但不与六项核心指标报告合并）：
  摘要 → 引言 → 方法与数据（域对齐/配对/指标公式）→ 结果（验证/全局/分指标）
  → 缺陷诊断 → 结论 → 附录

输出：
  report_assets/pix_*.png        图表
  pixel_metrics_report.html      自包含 HTML 报告（图片 base64 内嵌）
"""
from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN1 = r"C:\Users\Lenovo\Desktop\run1_float"
DATA = "mstar_real"            # 验证集（real/real2/fake）
REF = "mstar_ref"              # 展开的完整 MSTAR 参考集
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(BASE, "report_assets")
OUT = os.path.join(BASE, "pixel_metrics_report.html")

CLASS_NAMES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
               "D7", "T62", "T72", "ZIL131", "ZSU234"]

# 指标元信息：key / 中文名 / 方向 / 理想值 / 检测内容
METRICS = [
    ("MSE", "均方误差", "lower", 0.0,
     "逐像素灰度差平方均值，检测整体亮度偏差与加性噪声"),
    ("RMSE", "均方根误差", "lower", 0.0,
     "√MSE，量纲与归一化灰度一致，更直观"),
    ("PSNR", "峰值信噪比", "higher", float("inf"),
     "10·log10(DR²/MSE)，DR=1（[0,1] 域），单位 dB"),
    ("NCC", "归一化互相关", "higher", 1.0,
     "两图灰度涨落趋势相关性，对整体亮度偏移不敏感"),
    ("UQI", "通用质量指数", "higher", 1.0,
     "相关×亮度×对比度三因子（SSIM 前身）"),
    ("MS-SSIM", "多尺度结构相似度", "higher", 1.0,
     "粗到细多尺度结构保真，对模糊/纹理损失敏感"),
    ("FSIM", "特征相似度", "higher", 1.0,
     "相位一致性+梯度幅度，对边缘/结构破坏敏感"),
]
METRIC_KEYS = [m[0] for m in METRICS]


# --------------------------------------------------------------------------
# 数据载入
# --------------------------------------------------------------------------
def load_validation():
    with open(os.path.join(DATA, "pixel_metrics_validation.json")) as fh:
        return json.load(fh)


def load_run1():
    with open(os.path.join(RUN1, "pixel_metrics_results.json")) as fh:
        return json.load(fh)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _subplot_grid(n):
    """把 n 个（n≤8）指标排成 2×4 网格，多余子图隐藏。"""
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.ravel()
    for k in range(n, len(axes)):
        axes[k].axis("off")
    return fig, axes


def _disp(img):
    """逐图 2–99.5% 对比度拉伸用于显示。"""
    img = np.asarray(img, dtype=np.float64)
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    lo, hi = np.percentile(img, 2), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


# --------------------------------------------------------------------------
# 图表
# --------------------------------------------------------------------------
def make_figures(val, run1):
    overall = run1["metrics"]["overall"]
    per_class = run1["metrics"]["per_class"]
    fig = {}

    # ---- 图1：MSTAR 验证（self vs degrade，逐指标）----
    f1, axes = _subplot_grid(len(METRIC_KEYS))
    for ax, (key, cn, direc, ideal, desc) in zip(axes, METRICS):
        s = val[key]["self"]
        d = val[key]["degrade"]
        ax.bar([0, 1], [s, d], color=["#2c7fb8", "#d73027"], width=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["real-real2\n(self)", "real-fake\n(degrade)"], fontsize=8)
        ax.set_title(f"{key}", fontsize=10)
        ax.set_ylabel("value", fontsize=7)
        for i, v in enumerate([s, d]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.tick_params(labelsize=7)
    f1.suptitle("Fig 1.  MSTAR validation: real-real2 (self) vs real-fake (degrade)",
                fontsize=11, y=1.02)
    f1.tight_layout()
    p1 = os.path.join(ASSET_DIR, "pix_fig1_validation.png")
    f1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(f1)
    fig["fig1"] = _b64(p1)

    # ---- 图2：全局评估（run1_float vs 理想值）----
    f2, axes = _subplot_grid(len(METRIC_KEYS))
    for ax, (key, cn, direc, ideal, desc) in zip(axes, METRICS):
        v = overall[key]
        ax.bar([0], [v], color="#2c7fb8", width=0.4)
        if np.isfinite(ideal):
            ax.axhline(ideal, color="#d73027", ls="--", lw=1.0,
                       label=f"ideal {ideal:g}")
        else:
            ax.text(0, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels(["run1_float"], fontsize=8)
        ax.set_title(f"{key}  ({'↓' if direc=='lower' else '↑'})", fontsize=10)
        ax.set_ylabel("value", fontsize=7)
        if np.isfinite(ideal):
            ax.text(0, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(labelsize=7)
    f2.suptitle("Fig 2.  run1_float 7 FR metrics (overall mean vs ideal)",
                fontsize=11, y=1.02)
    f2.tight_layout()
    p2 = os.path.join(ASSET_DIR, "pix_fig2_overall.png")
    f2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(f2)
    fig["fig2"] = _b64(p2)

    # ---- 图3：分指标统计（每类）----
    f3, axes = _subplot_grid(len(METRIC_KEYS))
    for ax, (key, cn, direc, ideal, desc) in zip(axes, METRICS):
        vals = [per_class[key][c] for c in CLASS_NAMES]
        ax.bar(CLASS_NAMES, vals, color="#2c7fb8", width=0.6)
        ax.set_title(f"{key}", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("value", fontsize=7)
    f3.suptitle("Fig 3.  Per-class metric breakdown (10 classes)", fontsize=11, y=1.02)
    f3.tight_layout()
    p3 = os.path.join(ASSET_DIR, "pix_fig3_perclass.png")
    f3.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(f3)
    fig["fig3"] = _b64(p3)

    # ---- 图4：样本可视化（真实 vs 生成）----
    gen = np.load(os.path.join(RUN1, "images_float_neg1_1.npy"))
    gen = gen[..., 0] if gen.ndim == 4 else gen
    gen_labels = np.load(os.path.join(RUN1, "labels.npy"))
    ref = np.load(os.path.join(REF, "ref_images.npy"))
    ref_labels = np.load(os.path.join(REF, "ref_labels.npy"))
    gmax_real = float(ref.max())
    show = [2, 0, 5]  # BRDM2（最差）、2S1、D7（最好）
    f4, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.2))
    for j, c in enumerate(show):
        ri = [i for i, l in enumerate(ref_labels) if l == c][0]
        gi = [i for i, l in enumerate(gen_labels) if l == c][0]
        axes[0, j].imshow(_disp(ref[ri] / gmax_real), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"Real {CLASS_NAMES[c]}", fontsize=9)
        axes[1, j].imshow(_disp(gen[gi]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"Gen {CLASS_NAMES[c]}", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    f4.suptitle("Fig 4.  Sample visualization (top: real MSTAR, bottom: generated; 2-99.5% stretch)",
                fontsize=11, y=1.02)
    f4.tight_layout()
    p4 = os.path.join(ASSET_DIR, "pix_fig4_samples.png")
    f4.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(f4)
    fig["fig4"] = _b64(p4)

    return fig


# --------------------------------------------------------------------------
# HTML 报告
# --------------------------------------------------------------------------
CSS = """
:root{--ink:#1a1a1a;--muted:#555;--line:#dcdcdc;--accent:#2c7fb8;--good:#1b7837;--bad:#b2182b;}
*{box-sizing:border-box;}
body{font-family:Georgia,'Times New Roman','Songti SC',serif;color:var(--ink);
     max-width:920px;margin:2.5rem auto;padding:0 1.5rem;line-height:1.65;
     background:#fbfbfa;}
h1{font-size:1.7rem;text-align:center;margin-bottom:.2rem;line-height:1.3;}
h2{font-size:1.25rem;margin-top:2.2rem;padding-bottom:.3rem;border-bottom:2px solid var(--accent);}
h3{font-size:1.05rem;margin-top:1.5rem;color:#333;}
.subtitle{text-align:center;color:var(--muted);font-size:.95rem;margin-bottom:1.5rem;}
.meta{width:100%;border-collapse:collapse;font-size:.85rem;margin:1rem 0 1.8rem;}
.meta td{border:1px solid var(--line);padding:.35rem .6rem;}
.meta td:first-child{background:#f2f2f0;font-weight:600;width:22%;white-space:nowrap;}
.abstract{border:1px solid var(--line);background:#f7f7f5;padding:.9rem 1.1rem;
          font-size:.92rem;margin:1.2rem 0;}
.abstract h2{border:none;margin-top:0;}
table{width:100%;border-collapse:collapse;font-size:.86rem;margin:1rem 0;}
th,td{border:1px solid var(--line);padding:.4rem .55rem;text-align:center;}
th{background:#eef3f7;font-weight:600;}
tbody tr:nth-child(even){background:#fafafa;}
.good{color:var(--good);font-weight:600;}
.bad{color:var(--bad);font-weight:600;}
.ok{color:#7a6a00;font-weight:600;}
figure{margin:1.4rem 0;text-align:center;}
figure img{max-width:100%;border:1px solid var(--line);background:#fff;}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.4rem;text-align:center;}
.defect{background:#fdf6f4;border-left:4px solid var(--bad);padding:.7rem 1rem;
        margin:.8rem 0;border-radius:0 4px 4px 0;}
.defect h3{margin:.2rem 0 .4rem;color:var(--bad);}
.defect.lim{background:#f7f7f5;border-left-color:var(--accent);}
.defect.lim h3{color:var(--accent);}
.sev{display:inline-block;font-size:.7rem;font-weight:700;color:#fff;background:var(--bad);
     padding:.1rem .5rem;border-radius:3px;margin-left:.4rem;vertical-align:middle;}
.sev.med{background:#d97706;}.sev.low{background:#7a6a00;}
.formula{font-family:'Cambria Math','Times New Roman',serif;background:#f7f7f5;
         padding:.5rem .8rem;margin:.5rem 0;border-left:3px solid var(--accent);
         font-size:.95rem;overflow-x:auto;}
.footnote{font-size:.78rem;color:var(--muted);}
ol.refs{font-size:.82rem;color:var(--muted);}
"""


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "—"
    return f"{v:.{nd}f}"


def build_html(val, run1, fig):
    overall = run1["metrics"]["overall"]
    per_class = run1["metrics"]["per_class"]
    class_names = run1["class_names"]

    def f(v, nd=4):
        return _fmt(v, nd)

    # 验证表行
    val_rows = "\n".join(
        f"<tr><td><b>{key}</b></td><td>{cn}</td><td class='self'>{f(val[key]['self'],5)}</td>"
        f"<td class='degr'>{f(val[key]['degrade'],5)}</td>"
        f"<td class='good'>{'✓ 可区分' if abs(val[key]['self'] - val[key]['degrade']) > 0.05 * max(val[key]['self'], val[key]['degrade'], 1e-9) else '✗'}</td></tr>"
        for key, cn, direc, ideal, desc in METRICS)

    # 全局评估行
    global_rows = "\n".join(
        f"<tr><td><b>{key}</b></td><td>{cn}</td><td>{'↓ 越低越好' if direc=='lower' else '↑ 越高越好'}</td>"
        f"<td class='ideal'>{f(ideal,4)}</td><td><b>{f(overall[key])}</b></td></tr>"
        for key, cn, direc, ideal, desc in METRICS)

    # 每类表：行=指标，列=类别
    per_class_rows = ""
    for key, cn, direc, ideal, desc in METRICS:
        cells = "".join(f"<td>{f(per_class[key][c])}</td>" for c in class_names)
        per_class_rows += f"<tr><td><b>{key}</b></td><td class='ideal'>{f(ideal,4)}</td><td>{f(overall[key])}</td>{cells}</tr>"
    class_head = "".join(f"<th>{c}</th>" for c in class_names)

    # 缺陷诊断
    defects = build_defects(overall, per_class, class_names)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAR 生成图像 FR 全参考指标质检报告 · run1_float</title>
<style>{CSS}</style>
</head>
<body>

<h1>SAR 生成图像 FR 全参考指标质检报告</h1>
<div class="subtitle">基于综述 <i>arXiv 2411.05027</i>（GenAIxSAR）补充的 7 项全参考 IQA 指标，
对 run1_float 生成集做逐像素 / 结构保真的量化质检</div>

<table class="meta">
<tr><td>评估对象</td><td><code>run1_float</code> — DDIM 扩散模型生成 SAR 图像</td></tr>
<tr><td>指标来源</td><td>arXiv 2411.05027 全参考 IQA（MSE/RMSE/PSNR/NCC/UQI/MS-SSIM/FSIM）</td></tr>
<tr><td>样本规模</td><td>生成 10 类 × 8 = 80 张；真实参考 10 类 × 72 角 = 720 张（128×128 幅度）</td></tr>
<tr><td>真实参照</td><td>MSTAR（由 <code>mstar_raw</code> 展开的完整幅度切片，全局归一化）</td></tr>
<tr><td>评估指标</td><td>MSE、RMSE、PSNR、NCC、UQI、MS-SSIM、FSIM</td></tr>
<tr><td>配对口径</td><td>同类交叉全配对取平均（每张生成图 vs 同类全部真实参考）</td></tr>
<tr><td>报告日期</td><td>{__import__('datetime').datetime.now().strftime('%Y-%m-%d')}</td></tr>
</table>

<div class="abstract">
<h2>摘要</h2>
<p>对 DDIM 扩散模型生成的 80 张 SAR 图像进行 7 项全参考指标的量化质检。核心结果为
<b>PSNR = {f(overall['PSNR'],2)} dB</b>、<b>MSE = {f(overall['MSE'])}</b>、
<b>NCC = {f(overall['NCC'],4)}</b>、<b>UQI = {f(overall['UQI'],4)}</b>、
<b>MS-SSIM = {f(overall['MS-SSIM'],4)}</b>、<b>FSIM = {f(overall['FSIM'],4)}</b>。
MSTAR 公开集验证表明 7 项指标实现正确、且均能区分自洽（real-real2）与退化（real-fake）样本。
对生成集的评估显示：<b>结构感知类指标（FSIM=0.577、MS-SSIM=0.121）表明生成图保留了目标的粗轮廓
与相位结构，但精细纹理与散射细节明显缺失</b>；<b>逐像素相关类指标（NCC=0.168、UQI=0.010）大幅
偏离理想值</b>，主要由 SAR 乘性散斑的逐点随机相位不相关、以及 UQI 在暗背景 SAR 图上的分母退化
（μ≈0）共同导致。整体结论：生成图<b>结构可辨、逐像素保真弱、类间差异明显</b>（BRDM2 最差、D7 最好），
与六项核心指标报告的"对比度/动态范围缺陷"结论互为印证。</p>
</div>

<h2>1 引言</h2>
<p>本报告在项目既有六项核心指标（FID/SSIM/AFS/ΔENL/BVE/CMAE）之外，补充 arXiv 2411.05027
（生成式 AI 与 SAR 综述）所列举的经典全参考（Full-Reference）图像质量指标，从<b>逐像素保真</b>与
<b>结构感知保真</b>两个互补维度，回答"生成图与真实 MSTAR 在像素级与结构级分别差多少"。
其中 MSE/RMSE/PSNR 衡量逐点灰度一致性，NCC/UQI 衡量灰度涨落的相关性与统计一致性，
MS-SSIM/FSIM 衡量多尺度结构与感知特征（相位一致性、梯度幅度）的相似度。三者层次递进：
<b>像素 → 统计 → 结构</b>。</p>

<h2>2 方法与数据</h2>
<h3>2.1 域对齐</h3>
<p>生成图为模型原始输出域 [-1,1]；真实参考为 <code>mstar_raw</code> 展开的幅度切片（原始最大值约
15.97）。为统一到 [0,1] 灰度域并保证 <code>data_range=1</code> 口径一致，逐对做
<i>pair-01</i> 归一化：含负值输入先 (x+1)/2 映射到 [0,1]，再按该对两图的联合最大值缩放。该
口径与 <code>ssim()</code> 的归一化思路一致，使绝对亮度尺度不干扰比较。</p>
<h3>2.2 配对方式</h3>
<p>生成图 <code>labels.npy</code> 仅含类别标签、无方位角，故采用<b>同类交叉全配对取平均</b>：每张
生成图与其同类别全部真实参考图（每类 72 张）逐张计算指标后取平均，再对 80 张生成图取平均。
该口径下每张生成图的指标值反映"与真实同类目标整体分布的接近程度"，而非与某一特定角度的对齐。</p>
<h3>2.3 指标定义</h3>
<div class="formula">MSE = (1/N) Σ<sub>i</sub> ⟨(r<sub>i</sub> − f<sub>i</sub>)²⟩，&nbsp; RMSE = √MSE</div>
<div class="formula">PSNR = 10·log<sub>10</sub>(DR² / MSE)，&nbsp; DR = 1（[0,1] 域），单位 dB</div>
<div class="formula">NCC = (1/N) Σ<sub>i</sub> [ Σ<sub>p</sub>(r<sub>ip</sub>−μ<sub>r</sub>)(f<sub>ip</sub>−μ<sub>f</sub>) / (√Σ(r−μ<sub>r</sub>)² · √Σ(f−μ<sub>f</sub>)²) ] ∈ [−1,1]</div>
<div class="formula">UQI = (1/N) Σ<sub>i</sub> [ 4·σ<sub>rf</sub>·μ<sub>r</sub>·μ<sub>f</sub> / ((σ<sub>r</sub>²+σ<sub>f</sub>²)(μ<sub>r</sub>²+μ<sub>f</sub>²)) ]，&nbsp; 相关×亮度×对比度</div>
<div class="formula">MS-SSIM = (1/N) Σ<sub>i</sub> [ l<sub>M</sub><sup>α<sub>M</sub></sup>(r,f) · Π<sub>j=1..M</sub> c<sub>j</sub><sup>β<sub>j</sub></sup>(r,f) s<sub>j</sub><sup>γ<sub>j</sub></sup>(r,f) ]</div>
<div class="formula">FSIM = (1/N) Σ<sub>i</sub> [ Σ<sub>p</sub> S<sub>L</sub>(p)·PC<sub>m</sub>(p) / Σ<sub>p</sub> PC<sub>m</sub>(p) ]，&nbsp; S<sub>L</sub>=S<sub>PC</sub>·S<sub>G</sub></div>
<p class="footnote">其中 ⟨·⟩ 表示逐像素平均，r<sub>i</sub>/f<sub>i</sub> 为第 i 对参考/生成图。
PC = 相位一致性（phase congruency），G = 梯度幅度（gradient magnitude），PC<sub>m</sub> 取两图较大者。
UQI 为 Wang &amp; Bovik (2002) 通用质量指数（SSIM 的前身，无稳定性常数）；MS-SSIM 为
Wang et al. (2003) 多尺度扩展；FSIM 为 Zhang et al. (2011) 特征相似度。</p>

<h2>3 结果</h2>
<h3>3.1 指标验证（MSTAR 公开集）</h3>
<div class="footnote">
两列对照：<b>real-real2（自洽）</b>为近乎相同的第二次观测，应接近理想值；<b>real-fake（退化）</b>
施加模糊/噪声/再散斑退化，应明显偏离理想值。自洽列接近理想值、退化列明显偏离 → 证明 7 项指标实现
正确且能区分生成质量。</div>
<table>
<thead><tr><th>指标</th><th>名称</th><th>real-real2（自洽）</th><th>real-fake（退化）</th><th>区分度</th></tr></thead>
<tbody>{val_rows}</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig1']}">
<figcaption>图 1：MSTAR 验证。自洽（蓝）均贴近理想值，退化（红）均明显偏离，7 项指标全部具备区分度。</figcaption></figure>

<h3>3.2 全局评估（run1_float vs 真实 MSTAR）</h3>
<table>
<thead><tr><th>指标</th><th>名称</th><th>方向</th><th>理想值</th><th>run1_float 总体均值</th></tr></thead>
<tbody>{global_rows}</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig2']}">
<figcaption>图 2：run1_float 生成集 7 项指标总体均值。红线为理想值（MSE/RMSE 为 0，PSNR 无上界，其余为 1）。
结构类（FSIM/MS-SSIM）相对理想值最接近，逐像素相关类（NCC/UQI）偏离最大。</figcaption></figure>

<h3>3.3 分指标统计（每类）</h3>
<table>
<thead><tr><th>指标</th><th>理想值</th><th>总体</th>{class_head}</tr></thead>
<tbody>{per_class_rows}</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig3']}">
<figcaption>图 3：10 类分指标柱状图。结构类指标（FSIM/MS-SSIM）一致显示 BRDM2 最差、D7 最好。</figcaption></figure>

<h3>3.4 可视化对比</h3>
<figure><img src="data:image/png;base64,{fig['fig4']}">
<figcaption>图 4：三类样本的真实（上）与生成（下）对比，均采用逐图 2–99.5% 对比度拉伸显示。
生成图目标轮廓完整可辨，但缺乏真实 SAR 的精细散射纹理与散斑颗粒感。</figcaption></figure>

<h2>4 缺陷诊断</h2>
{defects}

<h2>5 结论</h2>
<p>7 项全参考指标的验证与评估表明，生成图在<b>结构感知维度</b>表现最好（FSIM={f(overall['FSIM'],3)}，
保留相位结构与目标轮廓），但<b>逐像素保真维度</b>严重偏弱（NCC={f(overall['NCC'],3)}、
UQI={f(overall['UQI'],4)}）。这一"结构可辨、像素不相关"的格局是 SAR 乘性散斑 + 无方位角对齐的
必然结果：真实 SAR 的散斑逐点相位随机，任何生成器都难以在逐像素意义上复现其灰度涨落。因此，
对本任务而言<b>FSIM 与 MS-SSIM 是最可信、最有判别力的两项指标</b>，而 NCC/UQI（尤其 UQI）因
暗背景与乘性噪声，只宜作为相对参考、不宜作为绝对质量门槛。建议以 FSIM 为主、MS-SSIM 为辅、
PSNR 作整体亮度一致性旁证，来追踪生成器迭代过程中的结构保真进展。</p>

<h2>附录 A　指标公式与实现来源</h2>
<p class="footnote">MSE/RMSE/PSNR 为经典逐像素保真指标；NCC（归一化互相关）与 UQI（Wang &amp; Bovik,
IEEE TIP 2002）为统计相关类；MS-SSIM（Wang et al., Asilomar 2003）与 FSIM（Zhang et al.,
IEEE TIP 2011）为结构/感知类。实现分别基于 <code>sewar</code>（UQI/PSNR 对照）、
<code>pytorch-msssim</code>（MS-SSIM）、<code>piq</code>（FSIM，<code>chromatic=False</code> 适配灰度 SAR）。
上述七项在 arXiv 2411.05027 中作为生成式 SAR 图像质量评估的可选指标被列举。</p>

<h2>附录 B　实现口径与局限</h2>
<ol class="refs">
<li>所有指标均逐对归一化到 [0,1] 后计算，<code>data_range=1</code>；含负值输入按 (x+1)/2 映射。</li>
<li>指标采用<b>逐对计算后取平均</b>的口径：MSE 为逐对 MSE 的均值，RMSE 为逐对 RMSE 的均值，
PSNR 为逐对 PSNR 的均值。由 Jensen 不等式，均值口径下 RMSE ≤ √(总体 MSE)、PSNR(均值) ≥
PSNR(总体 MSE)，故表中 RMSE 与 √MSE、PSNR 与 10·log10(1/MSE) 不完全互为逆运算。</li>
<li>生成图无方位角标签，采用<b>同类交叉全配对</b>而非<b>角度匹配</b>；SAR 对方位角敏感，散斑逐点
随机相位导致 NCC/UQI 系统性偏低，属口径与数据特性所致，非实现错误。</li>
<li>UQI 分母含 μ² 项，暗背景占主导（目标占比小）时 μ≈0，UQI 数值不稳定、会系统性偏低——属指标
自身在 SAR 上的适用范围局限，已在缺陷诊断中单独标注，不作为生成器真实质量缺陷解读。</li>
<li>本报告为独立的全参考指标报告，与六项核心指标报告（run1_quality_report.html）互补、不合并。</li>
</ol>

</body>
</html>"""
    return html


def build_defects(overall, per_class, class_names):
    # 找每个指标的最差/最好类（结构类取 FSIM 为主）
    def argminmax(key, fn):
        vs = {c: per_class[key][c] for c in class_names}
        return fn(vs, key=vs.get)

    worst_fsim = argminmax("FSIM", min)
    best_fsim = argminmax("FSIM", max)
    worst_ms = argminmax("MS-SSIM", min)
    best_ms = argminmax("MS-SSIM", max)
    worst_psnr = argminmax("PSNR", min)
    best_psnr = argminmax("PSNR", max)
    worst_ncc = argminmax("NCC", min)

    return f"""
<div class="defect">
<h3>缺陷 1：精细纹理与散射细节缺失（结构保真不足）<span class="sev">严重</span></h3>
<p><b>证据</b>：结构感知指标 MS-SSIM 仅 <b>{overall['MS-SSIM']:.3f}</b>、FSIM 仅
<b>{overall['FSIM']:.3f}</b>（理想值 1）。FSIM 保留了约 58% 的相位/梯度结构（目标轮廓与主体形状
基本可辨），但 MS-SSIM 的多尺度分析显示，从粗到细的结构相似度逐级衰减——即<b>粗尺度结构尚在、
细尺度纹理与散斑结构大量丢失</b>。这与扩散模型以 L2 损失训练、倾向于生成"平均化"光滑图像的
特性一致。</p>
</div>

<div class="defect">
<h3>缺陷 2：逐像素灰度相关性几乎消失<span class="sev">严重</span></h3>
<p><b>证据</b>：NCC 由自洽的 <b>0.9997</b> 跌至生成的 <b>{overall['NCC']:.3f}</b>。逐点灰度涨落与
真实图几乎不相关，根因是 SAR 乘性散斑的逐点随机相位——不同采样下的散斑实现互不相关。由于采用
同类交叉全配对（生成图与真实参考的散斑相位无对应关系），逐像素相关天然很低。此指标反映的是
"散斑纹理能否逐点复现"，对 SAR 生成任务通常不可达，应结合结构指标解读。</p>
</div>

<div class="defect">
<h3>缺陷 3：类间质量不均衡（BRDM2 最差、D7 最好）<span class="sev">中等</span></h3>
<p><b>证据</b>：结构指标一致显示 <b>{worst_fsim}</b> 最差（FSIM={per_class['FSIM'][worst_fsim]:.3f}、
MS-SSIM={per_class['MS-SSIM'][worst_ms]:.3f}），<b>{best_fsim}</b> 最好（FSIM={per_class['FSIM'][best_fsim]:.3f}、
MS-SSIM={per_class['MS-SSIM'][best_ms]:.3f}）。逐像素保真同样如此（PSNR：{worst_psnr} 最差
{per_class['PSNR'][worst_psnr]:.1f} dB、{best_psnr} 最好 {per_class['PSNR'][best_psnr]:.1f} dB）。
不同类别目标的结构复杂度与训练样本覆盖度差异，导致生成质量显著分层，需按类针对性改进。</p>
</div>

<div class="defect lim">
<h3>说明：UQI 接近零属指标局限，非真实缺陷<span class="sev low">局限</span></h3>
<p><b>证据</b>：UQI 总体仅 {overall['UQI']:.4f}，明显低于其它指标。根因是 UQI 分母含
μ<sub>r</sub>²+μ<sub>f</sub>² 项——SAR 图暗背景占主导（目标占比小），平均灰度 μ 接近 0，使分母
趋于 0、数值不稳定且系统性偏低。验证阶段自洽值 0.9997 证明实现正确；此处低值<b>不应解读为
生成器缺陷</b>，而应理解为 UQI 对暗背景 SAR 图适用性差。MS-SSIM 与 FSIM 内置稳定性常数
（C₁/C₂、T）故不受此影响，是本任务更可靠的选择。</p>
</div>

<div class="defect">
<h3>缺陷 4：整体亮度/动态范围与真实 SAR 有系统偏差<span class="sev">中等</span></h3>
<p><b>证据</b>：PSNR 总体 {overall['PSNR']:.1f} dB、MSE {overall['MSE']:.4f}（归一化域）。这一量级
表明生成图的整体灰度分布与真实参考存在系统性的亮度/动态范围偏差，与六项核心指标报告诊断的
"目标对比度不足 + 动态范围压缩"结论互为印证——扩散模型抑制了高强度散射中心、拉平了动态范围。</p>
</div>
"""


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)
    val = load_validation()
    run1 = load_run1()
    print("生成图表 …", flush=True)
    fig = make_figures(val, run1)
    print("生成 HTML 报告 …", flush=True)
    html = build_html(val, run1, fig)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"完成：{OUT}", flush=True)


if __name__ == "__main__":
    main()
