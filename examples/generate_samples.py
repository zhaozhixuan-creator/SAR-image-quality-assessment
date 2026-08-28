#!/usr/bin/env python3
"""生成仿真 SAR 样例图，用于端到端验证质检引擎。

生成两类样例：
- demo_single.tif : 单通道幅度图，注入点目标、扇贝、子带台阶、饱和、丢行、负值、边界填充
- demo_quad.tif   : 四极化幅度图（HH/HV/VH/VV），含互易性与通道不平衡
"""
import numpy as np
from PIL import Image


def speckle_scene(H=512, W=512, L=1, seed=0):
    """生成带纹理的强度场景（线性功率），乘性斑点。"""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    # 平滑纹理背景
    base = 1.0
    base += 0.6 * np.exp(-(((xx - W * 0.3) ** 2 + (yy - H * 0.4) ** 2) / (2 * 60 ** 2)))
    base += 0.5 * np.exp(-(((xx - W * 0.7) ** 2 + (yy - H * 0.7) ** 2) / (2 * 80 ** 2)))
    base += 0.15 * np.sin(2 * np.pi * xx / W * 3)
    # 乘性斑点：L 视 Gamma
    if L == 1:
        speckle = rng.exponential(1.0, size=(H, W))
    else:
        speckle = rng.gamma(L, 1.0 / L, size=(H, W))
    intensity = base * speckle
    return intensity


def add_point_target(intensity, y, x, peak=20000.0, width=7.0):
    """在强度域加截断 sinc² 点目标（非负，幅度旁瓣 ≈ -13 dB）。

    截断到 ±3 零点，避免 sinc² 的 1/x² 尾巴污染大范围（真实点响应有效旁瓣有限）。
    """
    H, W = intensity.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - x) / width
    dy = (yy - y) / width
    resp = np.sinc(dx) * np.sinc(dy)
    support = (np.abs(dx) < 3.0) & (np.abs(dy) < 3.0)
    resp = np.where(support, resp, 0.0)
    return intensity + peak * (resp ** 2)


def to_amp(intensity):
    return np.sqrt(np.maximum(intensity, 0.0))


def make_single():
    rng = np.random.RandomState(42)
    H = W = 512
    intensity = speckle_scene(H, W, L=4, seed=1)

    # 1) 点目标（峰值幅度 ≈ sqrt(122500) ≈ 350，未饱和、全图最亮孤立目标）
    intensity = add_point_target(intensity, 120, 360, peak=122500.0, width=7.0)
    amp = to_amp(intensity)

    # 2) 强散射"舰船"（用于饱和检测，随后被削平到上限）
    #    避免落在点目标附近（保护点目标 IRF 干净）
    n_ships = 2000
    sy = rng.randint(8, H - 8, n_ships)
    sx = rng.randint(8, W - 8, n_ships)
    too_close = (np.abs(sy - 120) < 30) & (np.abs(sx - 360) < 30)
    while too_close.any():
        sy[too_close] = rng.randint(8, H - 8, int(too_close.sum()))
        sx[too_close] = rng.randint(8, W - 8, int(too_close.sum()))
        too_close = (np.abs(sy - 120) < 30) & (np.abs(sx - 360) < 30)
    amp[sy, sx] = 700.0

    # 3) 扇贝效应：方位向周期性调制
    rows = np.arange(H)
    scallop = 1.0 + 0.08 * np.sin(2 * np.pi * rows / 128)
    amp = amp * scallop[:, None]

    # 4) 子带台阶：距离向列均值阶跃（约 1 dB）
    amp[:, 256:] *= 1.12

    # 5) 饱和削波（舰船 700 → 400；点目标 350 < 400 保留）
    amp = np.minimum(amp, 400.0)

    # 6) 丢行：一行置零
    amp[300, :] = 0.0

    # 7) 边界填充：四周置零
    amp[:4, :] = 0.0
    amp[-4:, :] = 0.0
    amp[:, :4] = 0.0
    amp[:, -4:] = 0.0

    # 8) 少量负值（模拟热噪声减除过减，仅内部区域，不污染边界填充）
    neg = np.zeros((H, W), dtype=bool)
    neg[8:H - 8, 8:W - 8] = rng.rand(H - 16, W - 16) < 0.002
    amp[neg] = -0.05 * rng.rand(int(neg.sum()))

    img = Image.fromarray(amp.astype(np.float32))
    img.save("demo_single.tif")
    print("已生成 demo_single.tif（单通道幅度，float32）")


def make_quad():
    H = W = 400
    intensity = speckle_scene(H, W, L=4, seed=2)
    intensity = add_point_target(intensity, 100, 300, peak=122500.0, width=7.0)
    amp = to_amp(intensity)
    rng = np.random.RandomState(7)
    HH = amp
    VV = amp * 0.95                          # 幅度不平衡 ≈ -0.45 dB
    HV = amp * 0.30 * (1 + 0.05 * rng.randn(H, W))
    VH = amp * 0.30 * (1 + 0.05 * rng.randn(H, W))   # 与 HV 统计一致（互易）
    quad = np.stack([HH, HV, VH, VV], axis=-1).astype(np.float32)
    # PIL 不支持多通道 float（RGBA float 会被静默截断为 uint8，破坏动态范围），
    # 用 tifffile 存多通道 float32 TIFF，io 层已回退支持读取。
    import tifffile
    tifffile.imwrite("demo_quad.tif", quad)
    print("已生成 demo_quad.tif（四极化 HH/HV/VH/VV，float32）")


if __name__ == "__main__":
    make_single()
    make_quad()
