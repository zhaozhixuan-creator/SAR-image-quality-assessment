from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .AASGconfig import AASGConfig, as_inject_alpha_map, build_aasg_config


def normalize_2nd_moment(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    # 输入: 任意张量 x。
    # 输出: 在指定维度上二阶矩归一化后的张量。
    # 作用: 稳定潜变量分布，避免幅值过大影响后续 MLP 学习。
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class W1Mapping(nn.Module):
    """将 z 与类别条件映射到不含角度语义的 w1。"""

    def __init__(self, z_dim: int, class_dim: int, w_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # 输入维度 = 随机噪声维度 + 类别条件维度。
        # 输出维度 = w1 维度（与主干 W 维度对齐）。
        in_dim = int(z_dim) + int(class_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, z: torch.Tensor, c_cls: torch.Tensor) -> torch.Tensor:
        # 输入:
        #   z: [N, z_dim]，随机噪声。
        #   c_cls: [N, class_dim]，类别 onehot（不含角度）。
        # 输出:
        #   w1: [N, w_dim]，角度无关的结构潜变量。
        # 作用:
        #   将“随机性 + 类别语义”编码成稀疏散射先验生成的驱动向量。
        z = normalize_2nd_moment(z.to(torch.float32))
        c_cls = c_cls.to(torch.float32)
        x = torch.cat([z, c_cls], dim=1)
        return self.net(x)


class SparseScatterPrior(nn.Module):
    """从 w1 预测稀疏散射点并软栅格化为 S0。"""

    def __init__(
        self,
        w_dim: int,
        k_points: int,
        base_resolution: int,
        sigma: float,
        hidden_dim: int,
        amp_eps: float,
        uv_radius_scale: float,
    ) -> None:
        super().__init__()
        # k_points: 稀疏点数量 K。
        # base_resolution: 基础稀疏图分辨率（例如 32x32）。
        # sigma: 高斯核宽度，控制每个散射点扩散范围。
        self.k_points = int(k_points)
        self.base_resolution = int(base_resolution)
        self.sigma = float(sigma)
        self.amp_eps = float(amp_eps)
        self.uv_radius_scale = float(uv_radius_scale)
        self.head = nn.Sequential(
            nn.Linear(w_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.k_points * 3),
        )

    def forward(self, w1: torch.Tensor, return_stats: bool = False, return_points: bool = False):
        # 输入:
        #   w1: [N, w_dim]，角度无关潜变量。
        # 输出:
        #   S0: [N, 1, H, W]，静态稀疏散射图。
        # 作用:
        #   将 w1 解码成 K 个散射点参数 (u,v,a)，再用高斯叠加生成稀疏先验。
        batch = w1.shape[0]
        points = self.head(w1).view(batch, self.k_points, 3)
        # 先用 tanh 映射到 [-1,1]，再乘半径缩放，硬性限制点位于中心区域。
        uv = torch.tanh(points[..., 0:2]) * self.uv_radius_scale
        amp = F.softplus(points[..., 2:3]) + self.amp_eps

        h = w = self.base_resolution
        ys = torch.linspace(-1.0, 1.0, h, device=w1.device, dtype=w1.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=w1.device, dtype=w1.dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)

        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, h, w, 2)

        delta = grid - uv.view(batch, self.k_points, 1, 1, 2)
        dist2 = (delta ** 2).sum(dim=-1, keepdim=True)
        sigma2 = max(self.sigma * self.sigma, 1e-8)
        s0 = (amp.view(batch, self.k_points, 1, 1, 1) * torch.exp(-dist2 / (2.0 * sigma2))).sum(dim=1)
        s0 = s0.permute(0, 3, 1, 2).contiguous()

        if (not return_stats) and (not return_points):
            return s0

        amp_flat = amp.view(batch, -1)
        s0_flat = s0.view(batch, -1)
        stats = {
            "amp_mean": float(amp_flat.mean().detach().cpu()),
            "amp_max": float(amp_flat.max().detach().cpu()),
            "amp_min": float(amp_flat.min().detach().cpu()),
            "uv_radius_mean": float(torch.sqrt((uv ** 2).sum(dim=2)).mean().detach().cpu()),
            "s0_mean": float(s0_flat.mean().detach().cpu()),
            "s0_max": float(s0_flat.max().detach().cpu()),
            "s0_nonzero_ratio": float((s0_flat > 1e-6).float().mean().detach().cpu()),
        }
        if return_stats and return_points:
            return s0, stats, uv, amp
        if return_stats:
            return s0, stats
        return s0, uv, amp


class AngleWarp(nn.Module):
    """将 S0 通过角度可微变换映射为 S(theta)。"""

    def __init__(self, class_dim: int, use_class_adaptive_scale: bool, scale_clamp: float) -> None:
        super().__init__()
        # base_scale: 全局可学习缩放参数。
        # class_to_scale: 类别相关缩放偏置（可选）。
        # 两者共同决定各向异性缩放，增强不同类别角度形变适配能力。
        self.use_class_adaptive_scale = bool(use_class_adaptive_scale)
        self.scale_clamp = float(scale_clamp)
        self.base_scale = nn.Parameter(torch.zeros(2))
        self.class_to_scale = nn.Linear(class_dim, 2) if self.use_class_adaptive_scale else None

    def forward(self, s0: torch.Tensor, c_cls: torch.Tensor, c_ang: torch.Tensor, return_stats: bool = False):
        # 输入:
        #   s0: [N, 1, H, W]，静态稀疏图。
        #   c_cls: [N, class_dim]，类别 onehot。
        #   c_ang: [N, 2]，角度分支 [sin(theta), cos(theta)]。
        # 输出:
        #   S_theta: [N, 1, H, W]，按角度对齐后的稀疏图。
        # 作用:
        #   用 sin/cos 构造旋转项，并结合类别自适应缩放形成仿射矩阵，
        #   再用 affine_grid + grid_sample 完成可微重采样。
        if c_ang.shape[1] != 2:
            raise ValueError(f"Expected c_ang dim=2 ([sin, cos]), got {c_ang.shape}")

        # Angle signal comes only from c_ang = [sin(theta), cos(theta)].
        sin_t = c_ang[:, 0]
        cos_t = c_ang[:, 1]
        norm = torch.sqrt((sin_t ** 2 + cos_t ** 2).clamp(min=1e-8))
        sin_t = sin_t / norm
        cos_t = cos_t / norm

        scale_delta = 0.0
        if self.class_to_scale is not None:
            scale_delta = self.class_to_scale(c_cls)
        scale_raw = self.base_scale.unsqueeze(0) + scale_delta
        scale_raw = torch.clamp(scale_raw, min=-self.scale_clamp, max=self.scale_clamp)
        sx, sy = torch.exp(scale_raw[:, 0]), torch.exp(scale_raw[:, 1])

        batch = s0.shape[0]
        theta = torch.zeros((batch, 2, 3), device=s0.device, dtype=s0.dtype)
        theta[:, 0, 0] = cos_t / sx
        theta[:, 0, 1] = -sin_t / sx
        theta[:, 1, 0] = sin_t / sy
        theta[:, 1, 1] = cos_t / sy

        grid = F.affine_grid(theta, size=s0.size(), align_corners=False)
        s_theta = F.grid_sample(s0, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if not return_stats:
            return s_theta

        st_flat = s_theta.view(s_theta.shape[0], -1)
        stats = {
            "stheta_mean": float(st_flat.mean().detach().cpu()),
            "stheta_max": float(st_flat.max().detach().cpu()),
            "stheta_nonzero_ratio": float((st_flat > 1e-6).float().mean().detach().cpu()),
            "sx_mean": float(sx.mean().detach().cpu()),
            "sy_mean": float(sy.mean().detach().cpu()),
        }
        return s_theta, stats


class MultiScalePriorBuilder(nn.Module):
    """将 S(theta) 变为多尺度先验，并提供各层固定注入权重。"""

    def __init__(self, inject_resolutions: Sequence[int], inject_alphas: Sequence[float]) -> None:
        super().__init__()
        # inject_resolutions: 需要注入的分辨率列表（中低层）。
        # inject_alphas: 各分辨率固定注入权重（建议递减）。
        if len(inject_resolutions) != len(inject_alphas):
            raise ValueError("inject_resolutions and inject_alphas must have the same length")
        self.inject_resolutions = tuple(int(r) for r in inject_resolutions)
        self.inject_alphas = tuple(float(a) for a in inject_alphas)

    def alpha_map(self) -> Dict[int, float]:
        # 输出:
        #   dict[resolution] = alpha，供主干注入时直接查询。
        return as_inject_alpha_map(self.inject_resolutions, self.inject_alphas)

    def forward(self, s_theta: torch.Tensor, target_resolutions: Optional[Iterable[int]] = None) -> Dict[int, torch.Tensor]:
        # 输入:
        #   s_theta: [N, 1, H, W]，角度对齐稀疏图。
        #   target_resolutions: 可选目标分辨率集合，不传则用配置默认值。
        # 输出:
        #   priors: {res: [N,1,res,res]}。
        # 作用:
        #   为主干不同分辨率层提供可直接注入的稀疏先验张量。
        resolutions = self.inject_resolutions if target_resolutions is None else tuple(int(r) for r in target_resolutions)
        out: Dict[int, torch.Tensor] = {}
        for res in resolutions:
            out[res] = F.interpolate(s_theta, size=(res, res), mode="bilinear", align_corners=False)
        return out


class AASGModule(nn.Module):
    """AASG skeleton: z + c -> w1 -> S0 -> S(theta) -> multiscale priors.

    Design constraint for this project:
    - w1 must be angle-free and only depend on z + class condition.
    - Angle control must come from c_ang=[sin, cos] through differentiable warp.
    """

    def __init__(self, z_dim: int, w_dim: int, cfg: Optional[AASGConfig] = None) -> None:
        super().__init__()
        # 输入:
        #   z_dim: 随机噪声维度。
        #   w_dim: w1 维度（通常与主干 W 维度一致）。
        #   cfg: AASG 配置。
        # 作用:
        #   组装 AASG 子模块：w1 映射、稀疏先验、角度变换、多尺度构建。
        self.z_dim = int(z_dim)
        self.w_dim = int(w_dim)
        self.cfg = build_aasg_config(cfg.to_dict()) if isinstance(cfg, AASGConfig) else build_aasg_config()

        self.mapping_w1 = W1Mapping(
            z_dim=self.z_dim,
            class_dim=self.cfg.class_dim,
            w_dim=self.w_dim,
            hidden_dim=self.cfg.mlp_hidden_dim,
        )
        self.prior = SparseScatterPrior(
            w_dim=self.w_dim,
            k_points=self.cfg.k_points,
            base_resolution=self.cfg.base_resolution,
            sigma=self.cfg.sigma,
            hidden_dim=self.cfg.mlp_hidden_dim,
            amp_eps=self.cfg.amp_eps,
            uv_radius_scale=self.cfg.uv_radius_scale,
        )
        self.warp = AngleWarp(
            class_dim=self.cfg.class_dim,
            use_class_adaptive_scale=self.cfg.use_class_adaptive_scale,
            scale_clamp=self.cfg.scale_clamp,
        )
        self.multiscale = MultiScalePriorBuilder(
            inject_resolutions=self.cfg.inject_resolutions,
            inject_alphas=self.cfg.inject_alphas,
        )

    def split_condition(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入:
        #   c: [N, C]，完整条件向量，约定格式为 [onehot..., sin, cos]。
        # 输出:
        #   c_cls: 类别分支（前 class_dim 维）。
        #   c_ang: 角度分支（从 cond_angle_start 起取 angle_dim 维）。
        # 作用:
        #   保证 w1 与角度语义解耦，角度仅进入可微变换分支。
        if c.ndim != 2:
            raise ValueError(f"Expected c shape [N, C], got {tuple(c.shape)}")
        min_dim = self.cfg.cond_angle_start + self.cfg.angle_dim
        if c.shape[1] < min_dim:
            raise ValueError(f"Expected c dim >= {min_dim}, got {c.shape[1]}")
        c_cls = c[:, : self.cfg.class_dim]
        c_ang = c[:, self.cfg.cond_angle_start : self.cfg.cond_angle_start + self.cfg.angle_dim]
        return c_cls, c_ang

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
        target_resolutions: Optional[Iterable[int]] = None,
        return_viz: bool = False,
    ) -> Dict[str, object]:
        # 输入:
        #   z: [N, z_dim]。
        #   c: [N, C]，完整标签（类别 + 角度）。
        #   target_resolutions: 可选注入分辨率。
        #   return_viz: 是否返回可视化中间量。
        # 输出:
        #   {
        #     "priors": {res: [N,1,res,res]},
        #     "alphas": {res: alpha},
        #     可选 "viz": {"S0": [N,1,H,W], "S_theta": [N,1,H,W]},
        #     可选 "w1": [N,w_dim]
        #   }
        # 作用:
        #   执行完整 AASG 流程: z+c_cls -> w1 -> S0 -> S(theta) -> 多尺度先验。
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected z shape [N, {self.z_dim}], got {tuple(z.shape)}")

        c_cls, c_ang = self.split_condition(c)
        w1 = self.mapping_w1(z=z, c_cls=c_cls)
        if return_viz:
            s0, s0_stats, uv, amp = self.prior(w1, return_stats=True, return_points=True)
            s_theta, st_stats = self.warp(s0=s0, c_cls=c_cls, c_ang=c_ang, return_stats=True)
        else:
            s0 = self.prior(w1)
            s_theta = self.warp(s0=s0, c_cls=c_cls, c_ang=c_ang)
        priors = self.multiscale(s_theta=s_theta, target_resolutions=target_resolutions)
        out: Dict[str, object] = {
            "priors": priors,
            "alphas": self.multiscale.alpha_map(),
        }
        if return_viz:
            stats = {}
            stats.update(s0_stats)
            stats.update(st_stats)
            out["viz"] = {"S0": s0, "S_theta": s_theta, "uv": uv, "amp": amp, "stats": stats}
            out["w1"] = w1
        return out
