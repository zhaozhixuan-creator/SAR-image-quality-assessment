from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .ENL_bgconfig import ENLBGConfig, build_enlbg_config


class ENLBGModule(nn.Module):
    """ENL_bg 模块：基于背景区域 ENL 统计的一致性约束。

    输入：
    - gen_img: [N, C, H, W]，生成图（训练域通常是 [-1, 1]）
    - real_img: [N, C, H, W]，真实图（训练域通常是 [-1, 1]）
    - cur_nimg: 当前已训练样本数（用于 warmup）

    输出：
    - loss: 标量张量，表示 ENL_bg 损失（已乘有效权重）
    - stats: 可选统计字典（用于日志观察）
    """

    def __init__(self, cfg: Optional[ENLBGConfig] = None) -> None:
        super().__init__()
        self.cfg = build_enlbg_config(cfg.to_dict()) if isinstance(cfg, ENLBGConfig) else build_enlbg_config()

        # 掩码缓存，避免每个 step 重复构造。
        # key: (H, W, device_str, dtype_str)
        self._mask_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}

    def _to_intensity(self, img: torch.Tensor) -> torch.Tensor:
        """将输入图转换为 [0,1] 强度图，并压到单通道。"""
        # 训练图像通常在 [-1,1]，先映射到 [0,1]
        x = (img.to(torch.float32) + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        if x.shape[1] == 1:
            return x

        # 当前仅支持通道均值灰度
        if self.cfg.intensity_mode == 'mean':
            return x.mean(dim=1, keepdim=True)

        # 理论上不会到这里（已在 config 校验）
        return x.mean(dim=1, keepdim=True)

    def _build_bg_mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构建背景掩码，背景为1，前景为0，形状 [1,1,H,W]。"""
        key = (h, w, str(device), str(dtype))
        if key in self._mask_cache:
            return self._mask_cache[key]

        mask = torch.ones((1, 1, h, w), device=device, dtype=dtype)
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0

        if self.cfg.mask_type == 'center_box':
            fh = max(1, int(round(h * self.cfg.fg_ratio)))
            fw = max(1, int(round(w * self.cfg.fg_ratio)))
            y0 = max(0, int(round(cy - fh / 2)))
            x0 = max(0, int(round(cx - fw / 2)))
            y1 = min(h, y0 + fh)
            x1 = min(w, x0 + fw)
            mask[:, :, y0:y1, x0:x1] = 0.0

        elif self.cfg.mask_type == 'center_circle':
            yy = torch.arange(h, device=device, dtype=dtype).view(h, 1)
            xx = torch.arange(w, device=device, dtype=dtype).view(1, w)
            r = float(min(h, w) * self.cfg.fg_ratio)
            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
            fg = dist2 <= (r ** 2)
            mask[:, :, fg] = 0.0

        self._mask_cache[key] = mask
        return mask

    def _compute_enl(self, intensity: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
        """计算每个样本的 ENL，返回形状 [N]。"""
        # 广播到 batch
        m = bg_mask
        if m.shape[0] == 1 and intensity.shape[0] > 1:
            m = m.expand(intensity.shape[0], -1, -1, -1)

        # 背景像素数量（每样本相同，但按 batch 形式保留）
        denom = m.sum(dim=[1, 2, 3]).clamp_min(1.0)

        mu = (intensity * m).sum(dim=[1, 2, 3]) / denom
        mu_b = mu.view(-1, 1, 1, 1)
        var = (((intensity - mu_b) ** 2) * m).sum(dim=[1, 2, 3]) / denom

        enl = (mu ** 2) / (var + self.cfg.eps_var)
        return enl

    def _lambda_eff(self, cur_nimg: Optional[int]) -> float:
        """计算 warmup 后的有效权重。"""
        base = float(self.cfg.weight)
        if self.cfg.warmup_kimg <= 0:
            return base
        if cur_nimg is None:
            return base
        warm = min(1.0, float(cur_nimg) / (self.cfg.warmup_kimg * 1000.0))
        return base * warm

    def forward(
        self,
        gen_img: torch.Tensor,
        real_img: torch.Tensor,
        cur_nimg: Optional[int] = None,
        return_stats: bool = False,
    ):
        """计算 ENL_bg 损失。"""
        if (not self.cfg.enabled) or self.cfg.weight <= 0:
            zero = gen_img.sum() * 0.0
            if return_stats:
                return zero, {
                    'enl_loss_raw': 0.0,
                    'enl_loss_weighted': 0.0,
                    'lambda_eff': 0.0,
                    'enl_gen_mean': 0.0,
                    'enl_real_mean': 0.0,
                }
            return zero

        gen_i = self._to_intensity(gen_img)
        real_i = self._to_intensity(real_img.detach())

        h, w = gen_i.shape[2], gen_i.shape[3]
        bg_mask = self._build_bg_mask(h=h, w=w, device=gen_i.device, dtype=gen_i.dtype)

        enl_gen = self._compute_enl(gen_i, bg_mask)
        enl_real = self._compute_enl(real_i, bg_mask)

        diff = torch.log(enl_gen + self.cfg.eps_log) - torch.log(enl_real + self.cfg.eps_log)
        enl_loss_raw = (diff ** 2).mean()

        lam = self._lambda_eff(cur_nimg)
        enl_loss = enl_loss_raw * lam

        if return_stats or self.cfg.debug:
            stats = {
                'enl_loss_raw': float(enl_loss_raw.detach().cpu()),
                'enl_loss_weighted': float(enl_loss.detach().cpu()),
                'lambda_eff': float(lam),
                'enl_gen_mean': float(enl_gen.mean().detach().cpu()),
                'enl_real_mean': float(enl_real.mean().detach().cpu()),
            }
            return enl_loss, stats

        return enl_loss
