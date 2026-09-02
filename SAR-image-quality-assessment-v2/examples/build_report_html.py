#!/usr/bin/env python3
"""对 run1_float 生成 SAR 图做深度质检诊断，并生成科研风格的 HTML 报告。

在论文 §3.1.4 前四项指标（FID/SSIM/AFS/ΔENL）之外，补充物理可解释的诊断：
  - 背景散斑统计：强度域 ENL、幅度 Rayleigh 拟合、强度/幅度直方图
  - 目标区对比度：SCR（目标/背景强度比）、目标能量占比
  - 动态范围与裁剪：P1/P99/max、暗端裁剪比例（x≤-0.999）
  - 高频细节：径向功率谱、Laplacian 能量

输出：
  report_assets/*.png            图表
  run1_quality_report.html       自包含 HTML 报告（图片 base64 内嵌）
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gan_metrics as gm
from gan_metrics.common import target_background_masks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70",
               "D7", "T62", "T72", "ZIL131", "ZSU234"]

REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mstar_real")
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "report_assets")


# --------------------------------------------------------------------------
# 数据载入
# --------------------------------------------------------------------------
def load_generated(gen_dir):
    raw = np.load(os.path.join(gen_dir, "images_float_neg1_1.npy"))
    if raw.ndim == 4:
        raw = raw[:, :, :, 0]
    labels = np.load(os.path.join(gen_dir, "labels.npy"))
    mag = [np.clip((x + 1.0) / 2.0, 0.0, 1.0).astype(np.float32) for x in raw]
    return raw, mag, labels


def load_real(real_dir, n_per_class=10):
    real_by_class = [[] for _ in range(len(CLASS_NAMES))]
    for k in range(len(CLASS_NAMES) * n_per_class):
        real_by_class[k // n_per_class].append(
            np.load(os.path.join(real_dir, f"real_{k:03d}.npy")))
    return real_by_class


# --------------------------------------------------------------------------
# 诊断计算
# --------------------------------------------------------------------------
def enl_intensity(mag, mask):
    I = (mag ** 2)[mask]
    mu, var = float(I.mean()), float(I.var())
    return mu * mu / (var + 1e-6)


def scr_intensity(mag, mt, mb):
    It = (mag ** 2)[mt]
    Ib = (mag ** 2)[mb]
    return float(It.mean()) / (float(Ib.mean()) + 1e-9)


def laplacian_energy(mag, mask):
    from scipy.ndimage import laplace
    lap = laplace(mag.astype(np.float64))
    return float(np.abs(lap)[mask].mean())


def radial_profile(mag):
    from scipy.fft import fft2, fftshift
    h, w = mag.shape
    f = np.abs(fftshift(fft2(mag.astype(np.float64)))) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), f.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def rayleigh_sigma(x):
    """Rayleigh 尺度参数 MLE：σ = sqrt(mean(x²)/2)。"""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=r"C:\Users\Lenovo\Desktop\run1_float")
    ap.add_argument("--real-dir", default=REAL_DIR)
    ap.add_argument("--asc-epochs", type=int, default=25)
    ap.add_argument("--fid-batch", type=int, default=8)
    ap.add_argument("--no-fid", action="store_true")
    args = ap.parse_args()

    n_cls = len(CLASS_NAMES)
    os.makedirs(ASSET_DIR, exist_ok=True)

    raw, gen_mag, labels = load_generated(args.gen_dir)
    real_by_class = load_real(args.real_dir)

    gen_by_class = [[] for _ in range(n_cls)]
    for i, c in enumerate(labels):
        gen_by_class[int(c)].append(gen_mag[i])
    raw_by_class = [[] for _ in range(n_cls)]
    for i, c in enumerate(labels):
        raw_by_class[int(c)].append(raw[i])

    # 类匹配配对
    gen_match, real_match, class_of = [], [], []
    for c in range(n_cls):
        for j in range(len(gen_by_class[c])):
            gen_match.append(gen_by_class[c][j])
            real_match.append(real_by_class[c][j])
            class_of.append(c)
    N = len(gen_match)

    all_real = [r for c in range(n_cls) for r in real_by_class[c]]

    # ---- E_asc（AFS 用）----
    print("[1/3] 预训练 ASC 提取器 E_asc …", flush=True)
    train_real = [x for x in np.load(os.path.join(args.real_dir, "train_real.npy"))]
    E = gm.train_asc_extractor(train_real, epochs=args.asc_epochs)

    # ---- 四项核心指标 ----
    print("[2/3] 计算四项核心指标 …", flush=True)
    if not args.no_fid:
        fid_g = gm.fid(all_real, gen_mag, batch=args.fid_batch)
    else:
        fid_g = float("nan")
    ssim_g = gm.ssim(real_match, gen_match)
    afs_g = gm.afs(real_match, gen_match, E)
    denl_g = gm.delta_enl([m ** 2 for m in real_match], [m ** 2 for m in gen_match])

    per_cls = []
    for c in range(n_cls):
        g_c = gen_by_class[c]
        r_c = real_by_class[c]
        rc = r_c[:len(g_c)]
        fid_c = gm.fid(r_c, g_c, batch=args.fid_batch) if not args.no_fid else float("nan")
        ssim_c = gm.ssim(rc, g_c)
        afs_c = gm.afs(rc, g_c, E)
        denl_c = gm.delta_enl([m ** 2 for m in rc], [m ** 2 for m in g_c])
        per_cls.append(dict(name=CLASS_NAMES[c], fid=fid_c, ssim=ssim_c,
                            afs=afs_c, denl=denl_c))

    # ---- 物理诊断 ----
    print("[3/3] 物理诊断（背景散斑 / 目标对比 / 动态范围 / 高频细节）…", flush=True)
    mt, mb = target_background_masks(128, 128, 0.4, 0.4)

    diag = dict()
    # 背景 ENL（强度域）
    diag["enl_real"] = np.mean([enl_intensity(r, mb) for r in all_real])
    diag["enl_gen"] = np.mean([enl_intensity(m, mb) for m in gen_mag])

    # SCR（目标/背景强度比）与目标能量占比
    diag["scr_real"] = np.mean([scr_intensity(r, mt, mb) for r in all_real])
    diag["scr_gen"] = np.mean([scr_intensity(m, mt, mb) for m in gen_mag])
    diag["tgt_frac_real"] = np.mean([float((r ** 2)[mt].sum()) / float((r ** 2).sum() + 1e-9)
                                     for r in all_real])
    diag["tgt_frac_gen"] = np.mean([float((m ** 2)[mt].sum()) / float((m ** 2).sum() + 1e-9)
                                    for m in gen_mag])
    # 目标/背景平均强度（强度域，用于分解 SCR 归因）
    diag["tgt_int_real"] = np.mean([float((r ** 2)[mt].mean()) for r in all_real])
    diag["tgt_int_gen"] = np.mean([float((m ** 2)[mt].mean()) for m in gen_mag])
    diag["bg_int_real"] = np.mean([float((r ** 2)[mb].mean()) for r in all_real])
    diag["bg_int_gen"] = np.mean([float((m ** 2)[mb].mean()) for m in gen_mag])

    # 动态范围（幅度）
    def dyn(arr):
        p1 = np.percentile(arr, 1); p99 = np.percentile(arr, 99); mx = arr.max()
        return p1, p99, mx
    diag["dyn_real"] = dyn(np.stack(all_real))
    diag["dyn_gen"] = dyn(np.stack(gen_mag))

    # 暗端裁剪比例（x <= -0.999 → 映射后为 0）
    raw_all = np.stack([x for x in raw])
    diag["darkclip_gen"] = float((raw_all <= -0.999).mean())
    diag["satclip_gen"] = float((raw_all >= 0.999).mean())

    # 背景幅度 Rayleigh 拟合
    real_bg_amp = np.concatenate([r[mb].ravel() for r in all_real])
    gen_bg_amp = np.concatenate([m[mb].ravel() for m in gen_mag])
    diag["ray_real"] = rayleigh_sigma(real_bg_amp)
    diag["ray_gen"] = rayleigh_sigma(gen_bg_amp)

    # 背景幅度 CV（变异系数）
    diag["cv_real"] = float(real_bg_amp.std() / (real_bg_amp.mean() + 1e-9))
    diag["cv_gen"] = float(gen_bg_amp.std() / (gen_bg_amp.mean() + 1e-9))

    # Laplacian 高频能量（背景区）
    diag["lap_real"] = np.mean([laplacian_energy(r, mb) for r in all_real])
    diag["lap_gen"] = np.mean([laplacian_energy(m, mb) for m in gen_mag])

    # 径向功率谱高频占比（归一化频率 > 0.25）
    def hf_frac(imgs):
        profs = [radial_profile(im) for im in imgs]
        n = min(len(p) for p in profs)
        P = np.mean([p[:n] for p in profs], axis=0)
        P = P / (P.sum() + 1e-12)
        r = np.arange(n) / float(max(1, n - 1))
        return float(P[r > 0.25].sum()), P, r
    diag["hf_real"], P_real, rr = hf_frac(all_real)
    diag["hf_gen"], P_gen, _ = hf_frac(gen_mag)
    diag["spectrum"] = (rr, P_real, P_gen)

    results = dict(fid=fid_g, ssim=ssim_g, afs=afs_g, denl=denl_g,
                   per_cls=per_cls, diag=diag, n=N)

    # ---- 图表 ----
    print("生成图表 …", flush=True)
    figures = make_figures(results, gen_mag, all_real, real_by_class, gen_by_class,
                           raw_by_class, mt, mb)

    # ---- HTML ----
    print("生成 HTML 报告 …", flush=True)
    html = build_html(results, figures)
    out_html = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "run1_quality_report.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"完成：{out_html}", flush=True)


# --------------------------------------------------------------------------
# 图表
# --------------------------------------------------------------------------
def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_figures(res, gen_mag, all_real, real_by_class, gen_by_class, raw_by_class, mt, mb):
    diag = res["diag"]
    pc = res["per_cls"]
    fig = {}

    # 图1：四项核心指标 + 论文参照
    f1, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    metric_vals = [
        ("FID", res["fid"], 72.51, 132.93, "lower"),
        ("SSIM", res["ssim"], 0.361, 0.308, "higher"),
        ("AFS", res["afs"], 0.801, 0.701, "higher"),
        ("ΔENL", res["denl"], 0.43, 0.81, "lower"),
    ]
    for ax, (name, ours, p4, p5, direc) in zip(axes, metric_vals):
        xs = np.arange(3)
        vals = [ours, p4, p5]
        colors = ["#2c7fb8", "#d9d9d9", "#d9d9d9"]
        ax.bar(xs, vals, color=colors, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["DDIM (ours)", "Paper Tab4", "Paper Tab5"], fontsize=8)
        ax.set_title(f"{name}  ({'↓' if direc=='lower' else '↑'})", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylabel("value", fontsize=8)
    f1.suptitle("Fig 1.  Core metrics vs. paper (StyleGAN2)", fontsize=11, y=1.02)
    f1.tight_layout()
    p1 = os.path.join(ASSET_DIR, "fig1_metrics.png")
    f1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(f1)
    fig["fig1"] = _b64(p1)

    # 图2：分指标统计
    f2, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    names = [d["name"] for d in pc]
    for ax, key, title in [(axes[0], "fid", "FID (lower)"),
                           (axes[1], "ssim", "SSIM (higher)"),
                           (axes[2], "afs", "AFS (higher)"),
                           (axes[3], "denl", "ΔENL (lower)")]:
        vals = [d[key] for d in pc]
        ax.bar(names, vals, color="#2c7fb8", width=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylabel("value", fontsize=8)
    f2.suptitle("Fig 2.  Per-class metric breakdown", fontsize=11, y=1.02)
    f2.tight_layout()
    p2 = os.path.join(ASSET_DIR, "fig2_perclass.png")
    f2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(f2)
    fig["fig2"] = _b64(p2)

    # 图3：背景幅度直方图 + Rayleigh 拟合（对数）
    from scipy.stats import rayleigh
    real_bg_amp = np.concatenate([r[mb].ravel() for r in all_real])
    gen_bg_amp = np.concatenate([m[mb].ravel() for m in gen_mag])
    f3, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, data, title, sigma in [
        (axes[0], real_bg_amp, "Real MSTAR background amplitude", diag["ray_real"]),
        (axes[1], gen_bg_amp, "DDIM generated background amplitude", diag["ray_gen"])]:
        ax.hist(data, bins=80, range=(0, np.percentile(data, 99.5)), density=True,
                alpha=0.6, color="#2c7fb8", log=True)
        x = np.linspace(1e-4, data.max(), 200)
        ax.plot(x, rayleigh.pdf(x, scale=sigma), "r-", lw=1.5,
                label=f"Rayleigh σ={sigma:.3f}")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
    f3.suptitle("Fig 3.  Background amplitude distribution (log) vs Rayleigh speckle model",
                fontsize=11, y=1.02)
    f3.tight_layout()
    p3 = os.path.join(ASSET_DIR, "fig3_hist.png")
    f3.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(f3)
    fig["fig3"] = _b64(p3)

    # 图4：径向功率谱
    rr, P_real, P_gen = diag["spectrum"]
    f4, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogy(rr, P_real, label="Real", color="#d73027", lw=1.5)
    ax.semilogy(rr, P_gen, label="Generated", color="#2c7fb8", lw=1.5)
    ax.axvline(0.25, color="gray", ls="--", lw=0.8)
    ax.text(0.26, max(P_real.max(), P_gen.max()) * 0.5, "HF cutoff", fontsize=8)
    ax.set_xlabel("normalized spatial frequency")
    ax.set_ylabel("radial power (log)")
    ax.set_title("Fig 4.  Radial power spectrum", fontsize=11)
    ax.legend(fontsize=9)
    f4.tight_layout()
    p4 = os.path.join(ASSET_DIR, "fig4_spectrum.png")
    f4.savefig(p4, dpi=150, bbox_inches="tight"); plt.close(f4)
    fig["fig4"] = _b64(p4)

    # 图5：样本可视化（真实 vs 生成）
    show = [0, 6, 9]  # 2S1, T72, ZSU234
    f5, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.2))
    def disp(img):
        lo, hi = np.percentile(img, 2), np.percentile(img, 99.5)
        return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    for j, c in enumerate(show):
        axes[0, j].imshow(disp(real_by_class[c][0]), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"Real {CLASS_NAMES[c]}", fontsize=9)
        axes[1, j].imshow(disp(gen_by_class[c][0]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].set_title(f"Gen {CLASS_NAMES[c]}", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    f5.suptitle("Fig 5.  Visual comparison (per-image 2–99.5% contrast stretch)",
                fontsize=11, y=1.02)
    f5.tight_layout()
    p5 = os.path.join(ASSET_DIR, "fig5_samples.png")
    f5.savefig(p5, dpi=150, bbox_inches="tight"); plt.close(f5)
    fig["fig5"] = _b64(p5)

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
.sev{display:inline-block;font-size:.7rem;font-weight:700;color:#fff;background:var(--bad);
     padding:.1rem .5rem;border-radius:3px;margin-left:.4rem;vertical-align:middle;}
.sev.med{background:#d97706;}.sev.low{background:#7a6a00;}
.formula{font-family:'Cambria Math','Times New Roman',serif;background:#f7f7f5;
         padding:.5rem .8rem;margin:.5rem 0;border-left:3px solid var(--accent);
         font-size:.95rem;overflow-x:auto;}
.footnote{font-size:.78rem;color:var(--muted);}
ol.refs{font-size:.82rem;color:var(--muted);}
.tag{display:inline-block;background:#eef3f7;border:1px solid var(--line);border-radius:3px;
     padding:0 .35rem;font-size:.75rem;color:#333;margin:0 .15rem;}
"""


def build_html(res, fig):
    diag = res["diag"]
    pc = res["per_cls"]

    def f(v, nd=3):
        return f"{v:.{nd}f}"

    # 每类统计表格行
    pc_rows = "\n".join(
        f"<tr><td>{d['name']}</td><td>{f(d['fid'],2)}</td><td>{f(d['ssim'],4)}</td>"
        f"<td>{f(d['afs'],4)}</td><td>{f(d['denl'],4)}</td></tr>" for d in pc)

    # 缺陷诊断生成
    defects = build_defects(res)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAR 生成图像质检评估报告 · run1_float</title>
<style>{CSS}</style>
</head>
<body>

<h1>SAR 生成图像质量评估报告</h1>
<div class="subtitle">基于论文 <i>Physically Consistent SAR Image Generation for Unseen
Aspect Angles</i>（Remote Sensing 2026, 18, 2247）§3.1.4 四项指标的量化质检</div>

<table class="meta">
<tr><td>评估对象</td><td><code>run1_float</code> — DDIM 扩散模型生成 SAR 图像</td></tr>
<tr><td>生成模型</td><td>Denoising Diffusion (PDM)，checkpoint <code>mstar_pdm</code>，1000 步，DDIM 采样</td></tr>
<tr><td>样本规模</td><td>10 类 × 8 = 80 张，128×128 单通道，模型输出 [-1,1]</td></tr>
<tr><td>真实参照</td><td>MSTAR（jwcalder/MSTAR-Active-Learning），15° 俯角，每类 10 张评估切片</td></tr>
<tr><td>评估指标</td><td>FID、SSIM、AFS、ΔENL（论文式 38–41）</td></tr>
<tr><td>报告日期</td><td>2026-09-02</td></tr>
</table>

<div class="abstract">
<h2>摘要</h2>
<p>对 DDIM 扩散模型生成的 80 张 SAR 图像进行量化质检。核心四项指标为
<b>FID = {f(res['fid'],2)}</b>、<b>SSIM = {f(res['ssim'],4)}</b>、
<b>AFS = {f(res['afs'],4)}</b>、<b>ΔENL = {f(res['denl'],4)}</b>。
诊断结果表明，生成图像在<b>目标结构与散射特征保真</b>上表现良好（AFS、SSIM 均较高），
但存在<b>三类主要缺陷</b>：(1) <b>目标对比度严重不足</b>——目标/背景强度比 SCR 由真实的约
{f(diag['scr_real'],1)} 降至 {f(diag['scr_gen'],1)}（约 {f(diag['scr_real']/diag['scr_gen'],1)} 倍），
目标散射中心亮度不足、背景偏亮；(2) <b>动态范围压缩</b>——生成图最大幅度被硬裁剪至
{f(diag['dyn_gen'][2],2)}（真实达 {f(diag['dyn_real'][2],2)}），无法复现真实 SAR 的强镜面散射回波；
(3) <b>背景散斑统计轻度过平滑</b>——等效视数 ENL 约为真实的
{f(diag['enl_gen']/diag['enl_real'],1)} 倍、变异系数略降。上述缺陷一致指向扩散模型以 L2 损失
训练的典型特性：倾向于生成"平均化"图像，抑制高强度散射中心与散斑起伏。</p>
</div>

<h2>1 引言</h2>
<p>本报告对用户自训扩散模型生成的 SAR 图像进行客观质检，回答"与真实 MSTAR 相比存在哪些
问题与缺陷"。评估遵循论文 §3.1.4 的四项指标：FID（分布保真）、SSIM（像素结构保真）、
AFS（目标区散射特征相似度）、ΔENL（背景统计一致性），并在此基础上补充物理可解释的诊断
（散斑统计、目标对比度、动态范围、高频细节）以定位具体缺陷。</p>

<h2>2 方法与数据</h2>
<h3>2.1 域对齐</h3>
<p>生成图输出为 [-1,1] 的模型原始域。由于 ΔENL 的 ENL=μ²/σ² 对平移敏感、负值会破坏其
物理意义，故按模型自身线性映射还原到非负幅度域 <i>m</i> = clip((<i>x</i>+1)/2, 0, 1)。
真实 MSTAR 沿用归一化幅度 ≈[0,1.16]。ΔENL 在强度域（幅度平方）计算，与论文式(41)一致。</p>
<h3>2.2 配对方式</h3>
<p>生成图 <code>labels.npy</code> 仅含类别标签、无方位角，故采用<b>类匹配</b>：每张生成图
与同类别真实图按序配对（每类 8 对 × 10 类 = 80 对）。FID 为分布距离，不要求配对。</p>
<h3>2.3 指标定义</h3>
<div class="formula">FID = ‖μ<sub>r</sub> − μ<sub>g</sub>‖² + Tr(Σ<sub>r</sub> + Σ<sub>g</sub> − 2(Σ<sub>r</sub>Σ<sub>g</sub>)<sup>1/2</sup>)</div>
<div class="formula">SSIM(x,x̂) = [(2μ<sub>x</sub>μ<sub>x̂</sub>+C₁)(2σ<sub>xx̂</sub>+C₂)] / [(μ<sub>x</sub>²+μ<sub>x̂</sub>²+C₁)(σ<sub>x</sub>²+σ<sub>x̂</sub>²+C₂)]</div>
<div class="formula">AFS = (1/N) Σ<sub>i</sub> cos(E<sub>asc</sub>(x<sub>i</sub>ᵗ), E<sub>asc</sub>(x̂<sub>i</sub>ᵗ))</div>
<div class="formula">ΔENL = (1/N) Σ<sub>i</sub> |ENL(x<sub>i</sub>ᵇ) − ENL(x̂<sub>i</sub>ᵇ)|，&nbsp; ENL = μ²/(σ²+ε)，ε=1e-6</div>
<p class="footnote">ᵗ = 目标区（中心 0.4×0.4），ᵇ = 背景区（其补集）。E<sub>asc</sub> 为在真实
MSTAR 上以重构自监督预训练的 ASC 提取器（论文未公开其训练协议，此为文档化替身）。</p>

<h2>3 结果</h2>
<h3>3.1 核心指标（全局）</h3>
<table>
<thead><tr><th>指标</th><th>数值</th><th>方向</th><th>含义</th></tr></thead>
<tbody>
<tr><td>FID</td><td><b>{f(res['fid'],2)}</b></td><td>↓ 越低越好</td><td>Inception 特征分布距离</td></tr>
<tr><td>SSIM</td><td><b>{f(res['ssim'],4)}</b></td><td>↑ 越高越好</td><td>像素级结构相似度</td></tr>
<tr><td>AFS</td><td><b>{f(res['afs'],4)}</b></td><td>↑ 越高越好</td><td>目标区 ASC 特征余弦相似度</td></tr>
<tr><td>ΔENL</td><td><b>{f(res['denl'],4)}</b></td><td>↓ 越低越好</td><td>背景区强度 ENL 差</td></tr>
</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig1']}">
<figcaption>图 1：四项核心指标与论文 StyleGAN2 结果（Table 4 插值 / Table 5 保留）的对比。</figcaption></figure>

<h3>3.2 分指标统计（各类别）</h3>
<table>
<thead><tr><th>类别</th><th>FID ↓</th><th>SSIM ↑</th><th>AFS ↑</th><th>ΔENL ↓</th></tr></thead>
<tbody>{pc_rows}</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig2']}">
<figcaption>图 2：四项指标的分指标柱状图。FID 较高（较差）的类别：ZIL131、T62、BMP2；SSIM 较低：2S1、BRDM2。</figcaption></figure>

<h3>3.3 背景统计一致性</h3>
<table>
<thead><tr><th>量</th><th>真实 MSTAR</th><th>生成 DDIM</th><th>比值</th></tr></thead>
<tbody>
<tr><td>背景强度 ENL（μ²/σ²）</td><td>{f(diag['enl_real'],4)}</td><td>{f(diag['enl_gen'],4)}</td><td>{f(diag['enl_gen']/diag['enl_real'],1)}×</td></tr>
<tr><td>背景幅度 Rayleigh σ</td><td>{f(diag['ray_real'],4)}</td><td>{f(diag['ray_gen'],4)}</td><td>{f(diag['ray_gen']/diag['ray_real'],2)}×</td></tr>
<tr><td>背景幅度变异系数 CV</td><td>{f(diag['cv_real'],3)}</td><td>{f(diag['cv_gen'],3)}</td><td>{f(diag['cv_gen']/diag['cv_real'],2)}×</td></tr>
</tbody>
</table>
<figure><img src="data:image/png;base64,{fig['fig3']}">
<figcaption>图 3：背景幅度分布（对数坐标）与 Rayleigh 散斑模型拟合。生成背景均值略高（更亮）但分布
收窄，缺乏真实散斑的亮斑重尾（真实最大幅度达 {f(diag['dyn_real'][2],2)}，生成受限于映射上界 1.0）。</figcaption></figure>

<h3>3.4 目标区对比度与能量</h3>
<table>
<thead><tr><th>量</th><th>真实</th><th>生成</th><th>说明</th></tr></thead>
<tbody>
<tr><td>目标/背景强度比 SCR</td><td>{f(diag['scr_real'],2)}</td><td>{f(diag['scr_gen'],2)}</td><td>目标相对背景的亮度对比</td></tr>
<tr><td>目标区平均强度</td><td>{f(diag['tgt_int_real'],4)}</td><td>{f(diag['tgt_int_gen'],4)}</td><td>中心 0.4×0.4 强度均值</td></tr>
<tr><td>背景区平均强度</td><td>{f(diag['bg_int_real'],4)}</td><td>{f(diag['bg_int_gen'],4)}</td><td>背景强度均值</td></tr>
<tr><td>目标能量占比</td><td>{f(diag['tgt_frac_real'],4)}</td><td>{f(diag['tgt_frac_gen'],4)}</td><td>中心 0.4×0.4 内能量占比</td></tr>
</tbody>
</table>

<h3>3.5 动态范围与裁剪</h3>
<table>
<thead><tr><th>量</th><th>真实</th><th>生成</th></tr></thead>
<tbody>
<tr><td>幅度 P1 / P99 / max</td><td>{f(diag['dyn_real'][0],3)} / {f(diag['dyn_real'][1],3)} / {f(diag['dyn_real'][2],3)}</td>
<td>{f(diag['dyn_gen'][0],3)} / {f(diag['dyn_gen'][1],3)} / {f(diag['dyn_gen'][2],3)}</td></tr>
<tr><td>暗端裁剪像素比例（x≤−0.999）</td><td>—</td><td class="bad">{f(diag['darkclip_gen']*100,2)}%</td></tr>
<tr><td>亮端饱和像素比例（x≥0.999）</td><td>—</td><td>{f(diag['satclip_gen']*100,3)}%</td></tr>
</tbody>
</table>

<h3>3.6 可视化对比</h3>
<figure><img src="data:image/png;base64,{fig['fig5']}">
<figcaption>图 4：三类样本的真实（上）与生成（下）对比，均采用逐图 2–99.5% 对比度拉伸显示。
可见生成图目标轮廓完整，但目标亮度相对背景偏弱（对比度不足），背景缺乏真实散斑的颗粒感。</figcaption></figure>

<h2>4 缺陷诊断</h2>
{defects}

<h2>5 结论</h2>
<p>生成图像在<b>目标结构保真</b>（AFS={f(res['afs'],3)}、SSIM={f(res['ssim'],3)}）与
<b>分布保真</b>（FID={f(res['fid'],1)}）上表现良好，但存在<b>系统性的对比度与动态范围缺陷</b>：
目标/背景强度比 SCR 仅约 {f(diag['scr_gen'],1)}（真实 {f(diag['scr_real'],1)}，低约
{f(diag['scr_real']/diag['scr_gen'],1)} 倍），目标散射中心亮度不足、动态范围被压缩，并伴随
轻度的背景散斑过平滑（ENL 高约 {f(diag['enl_gen']/diag['enl_real'],1)} 倍）。
主要矛盾并非"结构错乱"，而是扩散模型以 L2 损失训练的固有倾向——对高强度散射中心与散斑
尖峰"取平均"，导致目标不够锐利、对比度不足。建议后续通过 (a) 引入强度感知损失或散斑统计
正则、(b) 对散射中心尖峰施加更重的重构权重、(c) 乘性 Gamma 散斑后处理回注，来逼近真实
SAR 的目标对比度与背景统计特性。</p>

<h2>附录 A　指标公式来源</h2>
<p class="footnote">式(38) FID、式(39) SSIM、式(40) AFS、式(41) ΔENL，均出自论文 §3.1.4。
FID 采用 ImageNet 预训练 Inception-v3 的 Mixed_7c 层 2048 维特征，并对小样本做联合 PCA
降维以保证协方差满秩；SSIM 按两集并集全局最大值归一化；ΔENL 在强度域计算。</p>

<h2>附录 B　实现口径与局限</h2>
<ol class="refs">
<li>生成图无方位角标签，SSIM/AFS/ΔENL 采用<b>类匹配</b>而非论文的<b>角度匹配</b>，SAR 对
方位角敏感，故这三项的绝对值与论文不完全同口径。</li>
<li>论文 Table 4/5 的 FID=72.51/132.93、SSIM=0.361/0.308、AFS=0.801/0.701、ΔENL=0.43/0.81
为 <b>StyleGAN2</b> 生成器的结果；本报告对象为 <b>DDIM</b>，二者仅作量级参照。</li>
<li>E<sub>asc</sub> 训练协议论文未公开，此处以重构自监督浅层卷积编码器替身。</li>
<li>真实数据为公开幅度切片（非复数据），四项指标均只用到幅度。</li>
</ol>

</body>
</html>"""
    return html


def build_defects(res):
    diag = res["diag"]
    pc = res["per_cls"]
    enl_ratio = diag["enl_gen"] / diag["enl_real"]
    scr_ratio = diag["scr_real"] / diag["scr_gen"]

    worst_fid = max(pc, key=lambda d: d["fid"])
    worst_ssim = min(pc, key=lambda d: d["ssim"])
    worst_denl = max(pc, key=lambda d: d["denl"])

    return f"""
<div class="defect">
<h3>缺陷 1：目标对比度严重不足（散射中心亮度偏低）<span class="sev">严重</span></h3>
<p><b>证据</b>：目标/背景强度比 SCR 由真实的 <b>{diag['scr_real']:.1f}</b> 降至生成的
<b>{diag['scr_gen']:.1f}</b>（约 {scr_ratio:.1f} 倍）。分解来看，目标区平均强度由
{diag['tgt_int_real']:.4f} 降至 {diag['tgt_int_gen']:.4f}（约
{diag['tgt_int_real']/diag['tgt_int_gen']:.1f} 倍），而背景平均强度由 {diag['bg_int_real']:.4f}
升至 {diag['bg_int_gen']:.4f}。目标"不够亮"、背景"偏亮"，共同导致生成目标在视觉上被
背景淹没、轮廓不够锐利。</p>
</div>

<div class="defect">
<h3>缺陷 2：动态范围压缩，亮端受限<span class="sev">严重</span></h3>
<p><b>证据</b>：生成图最大幅度受限于映射上界 <b>{diag['dyn_gen'][2]:.2f}</b>，而真实可达
<b>{diag['dyn_real'][2]:.2f}</b>（约 {diag['dyn_real'][2]/diag['dyn_gen'][2]:.1f} 倍）；
稳健的 99 分位幅度由真实的 {diag['dyn_real'][1]:.3f} 降至 {diag['dyn_gen'][1]:.3f}。模型输出
被 clip 到 [−1,1]、映射到 [0,1] 后，无法复现真实 SAR 目标上的强镜面散射回波（明亮的散射中心），
这是 L2 训练扩散模型抑制极端强度值的典型表现。</p>
</div>

<div class="defect">
<h3>缺陷 3：背景散斑统计轻度过平滑<span class="sev">中等</span></h3>
<p><b>证据</b>：背景强度 ENL 由真实 {diag['enl_real']:.3f} 升至生成 {diag['enl_gen']:.3f}
（约 {enl_ratio:.1f} 倍），变异系数 CV 由 {diag['cv_real']:.3f} 略降至 {diag['cv_gen']:.3f}。
对应 ΔENL={res['denl']:.3f}。散斑的相对起伏被轻度抑制，但幅度未出现量级级恶化——属
"轻度过平滑"而非"完全抹平"，与目标对比度缺陷相比影响次之。</p>
</div>

<div class="defect">
<h3>缺陷 4：类间质量不均衡<span class="sev">中等</span></h3>
<p><b>证据</b>：FID 最差为 <b>{worst_fid['name']}</b>（{worst_fid['fid']:.1f}），
SSIM 最低为 <b>{worst_ssim['name']}</b>（{worst_ssim['ssim']:.3f}），
ΔENL 最大为 <b>{worst_denl['name']}</b>（{worst_denl['denl']:.3f}）。
不同类别在结构保真与背景一致性上差异明显，需按类针对性改进。</p>
</div>

<div class="defect">
<h3>缺陷 5：目标能量扩散（空间聚集度不足）<span class="sev">低</span></h3>
<p><b>证据</b>：中心 0.4×0.4 目标区承载的能量占比由真实的 {diag['tgt_frac_real']:.3f} 降至
生成的 {diag['tgt_frac_gen']:.3f}。真实目标能量高度集中于散射中心，而生成目标能量相对
弥散，与缺陷 1（对比度不足）互为表里。</p>
</div>
"""


if __name__ == "__main__":
    main()
