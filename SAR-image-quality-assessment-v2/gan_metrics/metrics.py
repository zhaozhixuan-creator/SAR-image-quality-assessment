"""论文 §3.1.4 的六项评估指标实现。

统一约定：real / fake 为图像集 list[np.ndarray]（每张 (H, W) 或 (H, W, C)）。
需要逐对比较的指标（SSIM / AFS / ΔENL / BVE / CMAE）按顺序配对，返回平均标量；
FID 只看两个集合的特征分布，不要求配对。
"""
from __future__ import annotations

import numpy as np

from .common import crop_target, target_background_masks, to_2d


# ---------- 背景统计（ENL / BVE） ----------
def enl(x_b: np.ndarray, eps: float = 1e-6) -> float:
    """ENL = μ² / (σ² + ε)，论文式(22)(41)，ε 固定 1e-6。"""
    x = to_2d(x_b).ravel()
    mu = float(x.mean())
    var = float(x.var())
    return mu * mu / (var + eps)


def delta_enl(real, fake, target_h: float = 0.4, target_w: float = 0.4,
              eps: float = 1e-6) -> float:
    """ΔENL = (1/N)Σ|ENL(x_b) − ENL(x̂_b)|，背景区（Mb = 1 − Mt）。"""
    vals = []
    for r, f in zip(real, fake):
        H, W = to_2d(r).shape
        _, mb = target_background_masks(H, W, target_h, target_w)
        vals.append(abs(enl(to_2d(r)[mb], eps) - enl(to_2d(f)[mb], eps)))
    return float(np.mean(vals))


def bve(real, fake, target_h: float = 0.4, target_w: float = 0.4) -> float:
    """BVE = (1/N)Σ|σ²(x_b) − σ²(x̂_b)|，背景区方差差，论文式(42)。"""
    vals = []
    for r, f in zip(real, fake):
        H, W = to_2d(r).shape
        _, mb = target_background_masks(H, W, target_h, target_w)
        vals.append(abs(float(to_2d(r)[mb].var()) - float(to_2d(f)[mb].var())))
    return float(np.mean(vals))


# ---------- SSIM ----------
def ssim(real, fake, data_range: float = 1.0, **kwargs) -> float:
    """SSIM（式 39），逐对计算后取平均。

    先按两集并集的全局最大值归一化到 [0,1]，再以 data_range=1 计算，
    避免「暗背景 + 大动态范围」导致 C1/C2 常数过大、SSIM 被抬高的偏差。
    """
    from skimage.metrics import structural_similarity

    gmax = float(max(np.max(to_2d(im)) for im in list(real) + list(fake)))
    gmax = max(gmax, 1e-9)
    vals = [structural_similarity(to_2d(r) / gmax, to_2d(f) / gmax,
                                   data_range=data_range, **kwargs)
            for r, f in zip(real, fake)]
    return float(np.mean(vals))


# ---------- AFS ----------
def afs(real, fake, extractor, target_h: float = 0.4, target_w: float = 0.4,
        size: int = 64, batch: int = 64, device: str = "cpu") -> float:
    """AFS = (1/N)Σ cos(E_asc(x_t), E_asc(x̂_t))，目标区 ASC 特征余弦相似度（式 40）。"""
    import torch
    import torch.nn.functional as F

    from .models import _to_tensor

    def feats(imgs):
        crops = [crop_target(im, target_h, target_w) for im in imgs]
        out = []
        for i in range(0, len(crops), batch):
            t = _to_tensor(crops[i:i + batch], size, normalize=True).to(device)
            with torch.no_grad():
                z = extractor(t)
            out.append(F.normalize(z, dim=1))
        return torch.cat(out, dim=0)

    zr = feats(real)
    zf = feats(fake)
    cos = (zr * zf).sum(1)
    return float(cos.mean())


# ---------- CMAE ----------
def cmae(estimated_angles, true_angles, period: float = 360.0) -> float:
    """CMAE = (1/N)Σ min(|φ̂−φ|, 360−|φ̂−φ|)，循环平均绝对误差（式 44），单位度。"""
    d = np.abs(np.asarray(estimated_angles, dtype=np.float64)
               - np.asarray(true_angles, dtype=np.float64))
    d = np.minimum(d, period - d)
    return float(np.mean(d))


def estimate_angles(images, estimator, size: int = 128, batch: int = 64,
                    device: str = "cpu") -> np.ndarray:
    """用方位角估计器 R 反推每张图的角度（度，范围 [0, 360)）。"""
    import torch

    from .models import _to_tensor

    out = []
    for i in range(0, len(images), batch):
        t = _to_tensor(images[i:i + batch], size, normalize=True).to(device)
        with torch.no_grad():
            sc = estimator(t).cpu().numpy()  # (N, 2): sin, cos
        ang = np.rad2deg(np.arctan2(sc[:, 0], sc[:, 1]))
        out.append(np.mod(ang, 360.0))
    return np.concatenate(out)


# ---------- FID ----------
def _joint_pca(fr, fg, d):
    """在联合集上做 PCA，把 fr/fg 投影到同一 d 维子空间（保证满秩协方差）。"""
    joint = np.concatenate([fr, fg], axis=0)
    mu = joint.mean(0, keepdims=True)
    c = joint - mu
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    Vd = Vt[:d].T  # (D, d)
    return (fr - mu) @ Vd, (fg - mu) @ Vd


def _frechet(mu1, sig1, mu2, sig2):
    """Fréchet 距离 = ‖μ1−μ2‖² + Tr(Σ1+Σ2−2(Σ1Σ2)^{1/2})。"""
    diff = mu1 - mu2
    dist = float(diff @ diff)
    from scipy.linalg import sqrtm

    covmean = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = np.clip(covmean, 0.0, None)
    val = dist + float(np.trace(sig1 + sig2 - 2.0 * covmean))
    return max(val, 0.0)


def fid(real, fake, feature_extractor=None, batch: int = 16, pca_dim: int = 64,
        size: int = 299, device: str = "cpu", global_max: float | None = None) -> float:
    """FID = ‖μr−μg‖² + Tr(Σr+Σg−2(ΣrΣg)^{1/2})，式(38)。

    特征先经 PCA 降维到 min(pca_dim, N−1) 以保证小样本协方差满秩
    （论文样本数 180 亦远小于 2048，此为文档化的数值稳定措施）。
    """
    from .inception import InceptionV3, preprocess

    if feature_extractor is None:
        feature_extractor = InceptionV3(pretrained=True).to(device)
        feature_extractor.eval()

    gmax = global_max if global_max is not None else \
        float(max(np.max(to_2d(im)) for im in list(real) + list(fake)))

    def feats(imgs):
        imgs = list(imgs)
        out = []
        for i in range(0, len(imgs), batch):
            t = preprocess(imgs[i:i + batch], size=size, global_max=gmax).to(device)
            out.append(feature_extractor(t).cpu().numpy())
        return np.concatenate(out, axis=0)

    fr = feats(real)
    fg = feats(fake)
    d = max(1, min(pca_dim, fr.shape[0] - 1, fg.shape[0] - 1))
    pr, pg = _joint_pca(fr, fg, d)
    mu_r, sig_r = pr.mean(0), np.cov(pr, rowvar=False)
    mu_g, sig_g = pg.mean(0), np.cov(pg, rowvar=False)
    return _frechet(mu_r, sig_r, mu_g, sig_g)


# ---------- 逐像素 / 结构 FR 指标（补充：arXiv 2411.05027 全参考 IQA）----------
def _pair_01(r, f):
    """逐对归一化到 [0,1] 幅度域。

    - 含负值的输入视为模型输出空间 [-1,1]，先 (x+1)/2 映射到 [0,1]（等价 png 灰度映射）；
    - 再按该对两图的联合最大值缩放，保证 data_range=1 口径一致（与 ssim() 同思路）。
    """
    r = to_2d(r).astype(np.float64)
    f = to_2d(f).astype(np.float64)
    if r.min() < 0.0:
        r = (r + 1.0) / 2.0
    if f.min() < 0.0:
        f = (f + 1.0) / 2.0
    gmax = max(float(r.max()), float(f.max()), 1e-9)
    return r / gmax, f / gmax


def mse(real, fake):
    """MSE = (1/N)Σ mean((r−f)²)，[0,1] 域逐对平均。理想 0。"""
    vals = []
    for r, f in zip(real, fake):
        a, b = _pair_01(r, f)
        vals.append(float(np.mean((a - b) ** 2)))
    return float(np.mean(vals))


def rmse(real, fake):
    """RMSE = √MSE，量纲与归一化灰度一致。理想 0。"""
    return float(np.sqrt(mse(real, fake)))


def psnr(real, fake, data_range: float = 1.0):
    """PSNR = 10·log10(DR²/MSE)，DR=1（[0,1] 域）。理想 +∞。"""
    m = mse(real, fake)
    if m <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range * data_range / m))


def ncc(real, fake):
    """NCC 归一化互相关 ∈ [−1,1]，1=完全线性相关。逐对平均。理想 1。"""
    vals = []
    for r, f in zip(real, fake):
        a, b = _pair_01(r, f)
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        den = float(np.sqrt((a @ a) * (b @ b)))
        vals.append(0.0 if den <= 0.0 else float((a @ b) / den))
    return float(np.mean(vals))


def uqi(real, fake):
    """UQI 通用质量指数（Wang&Bovik 2002）= 相关×亮度×对比度三因子。理想 1。"""
    vals = []
    eps = 1e-12
    for r, f in zip(real, fake):
        a, b = _pair_01(r, f)
        a = a.ravel()
        b = b.ravel()
        mu_a = float(a.mean())
        mu_b = float(b.mean())
        var_a = float(a.var())
        var_b = float(b.var())
        cov_ab = float(np.mean((a - mu_a) * (b - mu_b)))
        vals.append((4.0 * cov_ab * mu_a * mu_b)
                    / ((var_a + var_b + eps) * (mu_a ** 2 + mu_b ** 2 + eps)))
    return float(np.mean(vals))


def ms_ssim(real, fake, data_range: float = 1.0):
    """MS-SSIM 多尺度结构相似度（pytorch-msssim）。理想 1。"""
    import torch

    from pytorch_msssim import ssim as _ms_ssim

    vals = []
    for r, f in zip(real, fake):
        a, b = _pair_01(r, f)
        ta = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tb = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        vals.append(float(_ms_ssim(ta, tb, data_range=data_range, size_average=True)))
    return float(np.mean(vals))


def fsim(real, fake, data_range: float = 1.0):
    """FSIM 特征相似度（piq，相位一致性 + 梯度幅度）。理想 1。"""
    import torch

    from piq import fsim as _fsim

    vals = []
    for r, f in zip(real, fake):
        a, b = _pair_01(r, f)
        ta = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tb = torch.from_numpy(b.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        vals.append(float(_fsim(ta, tb, data_range=data_range, chromatic=False).item()))
    return float(np.mean(vals))
