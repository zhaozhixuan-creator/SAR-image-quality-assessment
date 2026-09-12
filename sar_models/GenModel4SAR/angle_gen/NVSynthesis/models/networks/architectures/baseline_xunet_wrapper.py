from collections import OrderedDict

import numpy as np
import torch
from einops import rearrange

from .XUNet_arch import (
    GroupNorm,
    ResnetBlock,
    XUNetBlock,
    posenc_ddpm,
    posenc_nerf,
)


class BaselinePinholeConditioningProcessor(torch.nn.Module):
    """Baseline optical-pinhole pose encoder adapted from reference_baseline/xunet.py."""

    def __init__(self, emb_ch, H, W, num_resolutions, use_pos_emb=True, use_ref_pose_emb=True):
        super().__init__()

        self.emb_ch = emb_ch
        self.num_resolutions = num_resolutions
        self.use_pos_emb = use_pos_emb
        self.use_ref_pose_emb = use_ref_pose_emb

        self.logsnr_emb_emb = torch.nn.Sequential(
            torch.nn.Linear(emb_ch, emb_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_ch, emb_ch),
        )

        D = 144
        if use_pos_emb:
            self.pos_emb = torch.nn.Parameter(torch.zeros(D, H, W), requires_grad=True)
            torch.nn.init.normal_(self.pos_emb, std=(1 / np.sqrt(D)))

        if use_ref_pose_emb:
            self.first_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.first_emb, std=(1 / np.sqrt(D)))
            self.other_emb = torch.nn.Parameter(torch.zeros(1, 1, D, 1, 1), requires_grad=True)
            torch.nn.init.normal_(self.other_emb, std=(1 / np.sqrt(D)))

        convs = []
        for i_level in range(self.num_resolutions):
            convs.append(
                torch.nn.Conv2d(
                    in_channels=D,
                    out_channels=self.emb_ch,
                    kernel_size=3,
                    stride=2 ** i_level,
                    padding=1,
                )
            )
        self.convs = torch.nn.ModuleList(convs)

    def forward(self, batch, cond_mask):
        B, C, H, W = batch["x"].shape

        logsnr = torch.clip(batch["logsnr"], -20, 20)
        logsnr_emb = posenc_ddpm(logsnr, emb_ch=self.emb_ch, max_time=1.0)
        logsnr_emb = self.logsnr_emb_emb(logsnr_emb)

        ray_pos, ray_dir = self._pinhole_rays(batch["R"], batch["t"], batch["K"], H, W, batch["x"])
        pose_emb_pos = posenc_nerf(ray_pos.float(), min_deg=0, max_deg=15)
        pose_emb_dir = posenc_nerf(ray_dir.float(), min_deg=0, max_deg=8)
        pose_emb = torch.concat([pose_emb_pos, pose_emb_dir], dim=-1)

        assert cond_mask.shape == (B,), (cond_mask.shape, B)
        cond_mask = cond_mask[:, None, None, None, None]
        pose_emb = torch.where(cond_mask, pose_emb, torch.zeros_like(pose_emb))
        pose_emb = rearrange(pose_emb, "b f h w c -> b f c h w")

        if self.use_pos_emb:
            pose_emb += self.pos_emb[None, None]
        if self.use_ref_pose_emb:
            pose_emb = torch.concat([self.first_emb, self.other_emb], axis=1) + pose_emb

        pose_embs = []
        for i_level in range(self.num_resolutions):
            B, F = pose_emb.shape[:2]
            pose_embs.append(
                rearrange(
                    self.convs[i_level](rearrange(pose_emb, "b f c h w -> (b f) c h w")),
                    "(b f) c h w -> b f c h w",
                    b=B,
                    f=F,
                )
            )

        return logsnr_emb, pose_embs

    @staticmethod
    def _expand_frames(value, frames):
        if value.ndim == 3 and value.shape[-2:] == (3, 3):
            return value[:, None].expand(-1, frames, -1, -1)
        if value.ndim == 2 and value.shape[-1] == 3:
            return value[:, None].expand(-1, frames, -1)
        return value

    def _pinhole_rays(self, R, t, K, H, W, reference):
        dtype = reference.dtype
        device = reference.device
        B = reference.shape[0]
        F = 2

        R = self._expand_frames(R.to(device=device, dtype=dtype), F)
        t = self._expand_frames(t.to(device=device, dtype=dtype), F)
        K = self._expand_frames(K.to(device=device, dtype=dtype), F)

        assert R.shape == (B, F, 3, 3), R.shape
        assert t.shape == (B, F, 3), t.shape
        assert K.shape == (B, F, 3, 3), K.shape

        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)

        K_inv = torch.linalg.inv(K)
        cam_dirs = torch.einsum("bfij,hwj->bfhwi", K_inv, pixels)
        cam_dirs = torch.nn.functional.normalize(cam_dirs, dim=-1, eps=1e-8)

        ray_dir = torch.einsum("bfij,bfhwj->bfhwi", R, cam_dirs)
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1, eps=1e-8)
        ray_pos = t[:, :, None, None, :].expand(-1, -1, H, W, -1)

        return ray_pos, ray_dir


class BaselineXUNetWrapper(torch.nn.Module):
    """
    Wrapper for the original reference_baseline XUNet.

    The original code consumes optical-camera ``R/t/K`` and RGB tensors.  This
    wrapper accepts the current SAR batch interface
    ``x/z/logsnr/azimuth_angle/incidence_angle`` and internally constructs a
    pinhole-camera surrogate, while preserving the original XUNet parameter names
    for checkpoint reuse.
    """

    inputH: int = 128
    inputW: int = 128
    ch: int = 256
    ch_mult: tuple[int] = (1, 2, 2, 4)
    emb_ch: int = 1024
    num_res_blocks: int = 3
    attn_resolutions: tuple[int] = (2, 3, 4)
    attn_heads: int = 4
    dropout: float = 0.1
    use_pos_emb: bool = True
    use_ref_pose_emb: bool = True
    pose_radius: float = 1.0
    pinhole_focal: float = 1.0
    reduce_rgb_output: str = "mean"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()

        assert self.inputH % (2 ** (len(self.ch_mult) - 1)) == 0, (
            f"Size of the image must be multiple of {2 ** (len(self.ch_mult) - 1)}"
        )
        assert self.inputW % (2 ** (len(self.ch_mult) - 1)) == 0, (
            f"Size of the image must be multiple of {2 ** (len(self.ch_mult) - 1)}"
        )

        self.num_resolutions = len(self.ch_mult)
        self.conditioningprocessor = BaselinePinholeConditioningProcessor(
            emb_ch=self.emb_ch,
            num_resolutions=self.num_resolutions,
            use_pos_emb=self.use_pos_emb,
            use_ref_pose_emb=self.use_ref_pose_emb,
            H=self.inputH,
            W=self.inputW,
        )

        self.conv = torch.nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding="same")

        self.dim_in = [self.ch] + (self.ch * np.array(self.ch_mult)[:-1]).tolist()
        self.dim_out = (self.ch * np.array(self.ch_mult)).tolist()

        self.xunetblocks = torch.nn.ModuleList([])
        for i_level in range(self.num_resolutions):
            single_level = torch.nn.ModuleList([])
            for i_block in range(self.num_res_blocks):
                use_attn = i_level in self.attn_resolutions
                single_level.append(
                    XUNetBlock(
                        in_channels=self.dim_in[i_level] if i_block == 0 else self.dim_out[i_level],
                        features=self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn,
                    )
                )

            if i_level != self.num_resolutions - 1:
                single_level.append(
                    ResnetBlock(
                        in_features=self.dim_out[i_level],
                        out_features=self.dim_out[i_level],
                        dropout=self.dropout,
                        resample="down",
                    )
                )
            self.xunetblocks.append(single_level)

        self.middle = XUNetBlock(
            in_channels=self.dim_out[-1],
            features=self.dim_out[-1],
            dropout=self.dropout,
            attn_heads=self.attn_heads,
            use_attn=self.num_resolutions in self.attn_resolutions,
        )

        self.upsample = torch.nn.ModuleDict()
        for i_level in reversed(range(self.num_resolutions)):
            single_level = torch.nn.ModuleList([])
            use_attn = i_level in self.attn_resolutions

            for i_block in range(self.num_res_blocks + 1):
                if i_block == 0:
                    prev_h_channels = (
                        self.dim_out[i_level + 1]
                        if (i_level + 1 < len(self.dim_out))
                        else self.dim_out[i_level]
                    )
                    prev_emb_channels = self.dim_out[i_level]
                elif i_block == self.num_res_blocks:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_in[i_level]
                else:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_out[i_level]

                single_level.append(
                    XUNetBlock(
                        in_channels=prev_h_channels + prev_emb_channels,
                        features=self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn,
                    )
                )

            if i_level != 0:
                single_level.append(
                    ResnetBlock(
                        in_features=self.dim_out[i_level],
                        out_features=self.dim_out[i_level],
                        dropout=self.dropout,
                        resample="up",
                    )
                )
            self.upsample[str(i_level)] = single_level

        self.lastgn = GroupNorm(num_channels=self.ch)
        self.lastconv = torch.nn.Conv2d(
            in_channels=self.ch, out_channels=3, kernel_size=3, stride=1, padding="same"
        )
        torch.nn.init.zeros_(self.lastconv.weight)

    def forward(self, batch, *, cond_mask):
        original_channels = batch["x"].shape[1]
        baseline_batch = self._to_baseline_batch(batch)
        rgb_noise = self._forward_baseline(baseline_batch, cond_mask=cond_mask)
        return self._from_baseline_output(rgb_noise, original_channels)

    def _forward_baseline(self, batch, *, cond_mask):
        B, C, H, W = batch["x"].shape

        for key, temp in batch.items():
            assert temp.shape[0] == B, f"{key} should have batch size of {B}, not {temp.shape[0]}"
        assert B == cond_mask.shape[0]
        assert (H, W) == (self.inputH, self.inputW), ((H, W), (self.inputH, self.inputW))

        logsnr_emb, pose_embs = self.conditioningprocessor(batch, cond_mask)

        h = torch.stack([batch["x"], batch["z"]], dim=1)
        h = self.conv(rearrange(h, "b f c h w -> (b f) c h w"))
        h = rearrange(h, "(b f) c h w -> b f c h w", b=B, f=2)

        hs = [h]
        for i_level in range(self.num_resolutions):
            emb = logsnr_emb[..., None, None] + pose_embs[i_level]

            for i_block in range(self.num_res_blocks):
                h = self.xunetblocks[i_level][i_block](h, emb)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = self.xunetblocks[i_level][-1](h, emb)
                hs.append(h)

        emb = logsnr_emb[..., None, None] + pose_embs[-1]
        h = self.middle(h, emb)

        for i_level in reversed(range(self.num_resolutions)):
            emb = logsnr_emb[..., None, None] + pose_embs[i_level]

            for i_block in range(self.num_res_blocks + 1):
                h = torch.concat([h, hs.pop()], dim=-3)
                h = self.upsample[str(i_level)][i_block](h, emb)

            if i_level != 0:
                h = self.upsample[str(i_level)][-1](h, emb)

        assert not hs

        h = torch.nn.functional.silu(self.lastgn(h))
        return rearrange(
            self.lastconv(rearrange(h, "b f c h w -> (b f) c h w")),
            "(b f) c h w -> b f c h w",
            b=B,
        )[:, 1]

    def _to_baseline_batch(self, batch):
        x = self._to_rgb(batch["x"])
        z = self._to_rgb(batch["z"])

        if {"R", "t", "K"}.issubset(batch.keys()):
            R, t, K = batch["R"], batch["t"], batch["K"]
        else:
            R, t, K = self._angles_to_pinhole_pose(
                batch["azimuth_angle"],
                batch["incidence_angle"],
                H=batch["x"].shape[-2],
                W=batch["x"].shape[-1],
                reference=batch["x"],
            )

        return {
            "x": x,
            "z": z,
            "logsnr": batch["logsnr"],
            "R": R,
            "t": t,
            "K": K,
        }

    @staticmethod
    def _to_rgb(img):
        if img.shape[1] == 3:
            return img
        if img.shape[1] == 1:
            return img.repeat(1, 3, 1, 1)
        raise ValueError(f"BaselineXUNetWrapper only supports 1 or 3 channels, got {img.shape[1]}")

    def _from_baseline_output(self, rgb_noise, original_channels):
        if original_channels == 3:
            return rgb_noise
        if original_channels != 1:
            raise ValueError(f"Cannot convert baseline output to {original_channels} channels.")
        if self.reduce_rgb_output == "first":
            return rgb_noise[:, :1]
        if self.reduce_rgb_output == "mean":
            return rgb_noise.mean(dim=1, keepdim=True)
        raise ValueError(f"Unsupported reduce_rgb_output={self.reduce_rgb_output}")

    def _angles_to_pinhole_pose(self, azimuth_angle, incidence_angle, H, W, reference):
        dtype = reference.dtype
        device = reference.device
        azimuth_angle = self._clean_angle(azimuth_angle).to(device=device, dtype=dtype)
        incidence_angle = self._clean_angle(incidence_angle).to(device=device, dtype=dtype)

        cos_azi = torch.cos(azimuth_angle)
        sin_azi = torch.sin(azimuth_angle)
        cos_inc = torch.cos(incidence_angle)
        sin_inc = torch.sin(incidence_angle)

        zeros_azi = torch.zeros_like(azimuth_angle)
        zeros_inc = torch.zeros_like(incidence_angle)
        ones_azi = torch.ones_like(azimuth_angle)
        ones_inc = torch.ones_like(incidence_angle)

        Rz = torch.stack(
            [
                torch.stack([cos_azi, -sin_azi, zeros_azi], dim=-1),
                torch.stack([sin_azi, cos_azi, zeros_azi], dim=-1),
                torch.stack([zeros_azi, zeros_azi, ones_azi], dim=-1),
            ],
            dim=-2,
        )
        Rx = torch.stack(
            [
                torch.stack([ones_inc, zeros_inc, zeros_inc], dim=-1),
                torch.stack([zeros_inc, cos_inc, -sin_inc], dim=-1),
                torch.stack([zeros_inc, sin_inc, cos_inc], dim=-1),
            ],
            dim=-2,
        )
        sar_R = torch.matmul(Rz, Rx)
        sar_forward = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        forward = torch.einsum("bfij,j->bfi", sar_R, sar_forward)
        forward = torch.nn.functional.normalize(forward, dim=-1, eps=1e-8)

        world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(forward)
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(forward)
        near_parallel = torch.abs((forward * world_up).sum(dim=-1, keepdim=True)) > 0.98
        up_seed = torch.where(near_parallel, alt_up, world_up)

        right = torch.cross(up_seed, forward, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1, eps=1e-8)
        up = torch.cross(forward, right, dim=-1)
        up = torch.nn.functional.normalize(up, dim=-1, eps=1e-8)

        R = torch.stack([right, up, forward], dim=-1)
        t = -float(self.pose_radius) * forward

        B = azimuth_angle.shape[0]
        focal = float(self.pinhole_focal) * max(H - 1, W - 1) / 2.0
        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 0] = focal
        K[:, 1, 1] = focal
        K[:, 0, 2] = (W - 1) / 2.0
        K[:, 1, 2] = (H - 1) / 2.0
        K[:, 2, 2] = 1.0

        return R, t, K

    @staticmethod
    def _clean_angle(angle):
        if angle.ndim == 3 and angle.shape[-1] == 1:
            angle = angle.squeeze(-1)
        if angle.ndim != 2:
            raise ValueError(f"Expected angle tensor with shape [B, 2], got {angle.shape}")
        return angle

    def load_checkpoint_state(self, checkpoint_state, strict=False, logger=None):
        target_state = self.state_dict()
        source_state = self._normalize_checkpoint_keys(checkpoint_state)
        load_state = OrderedDict()
        adapted = []
        skipped = []
        unexpected = []

        for key, tensor in source_state.items():
            if key not in target_state:
                unexpected.append(key)
                continue

            target_tensor = target_state[key]
            if tensor.shape == target_tensor.shape:
                load_state[key] = tensor
                continue

            converted = self._adapt_checkpoint_tensor(key, tensor, target_tensor)
            if converted is None:
                skipped.append((key, tuple(tensor.shape), tuple(target_tensor.shape)))
            else:
                load_state[key] = converted
                adapted.append((key, tuple(tensor.shape), tuple(target_tensor.shape)))

        missing, unexpected_after_load = self.load_state_dict(load_state, strict=False)
        self._log_checkpoint_report(
            logger=logger,
            loaded=len(load_state),
            adapted=adapted,
            skipped=skipped,
            missing=missing,
            unexpected=unexpected + list(unexpected_after_load),
        )
        if strict and (missing or unexpected_after_load or skipped or unexpected):
            raise RuntimeError("Baseline checkpoint loading finished with incompatible keys.")

    @staticmethod
    def _normalize_checkpoint_keys(checkpoint_state):
        normalized = OrderedDict()
        for key, value in checkpoint_state.items():
            clean_key = key
            if clean_key.startswith("module."):
                clean_key = clean_key[len("module.") :]
            if clean_key.startswith("model."):
                clean_key = clean_key[len("model.") :]
            normalized[clean_key] = value
        return normalized

    @staticmethod
    def _adapt_checkpoint_tensor(key, tensor, target_tensor):
        if key == "conv.weight" and tensor.ndim == 4 and target_tensor.ndim == 4:
            if tensor.shape[1] == 1 and target_tensor.shape[1] == 3 and tensor.shape[0] == target_tensor.shape[0]:
                return tensor.repeat(1, 3, 1, 1) / 3.0

        if key == "lastconv.weight" and tensor.ndim == 4 and target_tensor.ndim == 4:
            if tensor.shape[0] == 1 and target_tensor.shape[0] == 3 and tensor.shape[1:] == target_tensor.shape[1:]:
                return tensor.repeat(3, 1, 1, 1)

        if key == "lastconv.bias" and tensor.ndim == 1 and target_tensor.ndim == 1:
            if tensor.shape[0] == 1 and target_tensor.shape[0] == 3:
                return tensor.repeat(3)

        return None

    def _log_checkpoint_report(self, logger, loaded, adapted, skipped, missing, unexpected):
        pose_loaded = [
            key
            for key in self._normalize_checkpoint_keys(self.state_dict()).keys()
            if key.startswith("conditioningprocessor.")
        ]
        self._emit(logger, f"Baseline checkpoint filter loaded {loaded} tensors.")
        if adapted:
            self._emit(logger, "Adapted channel-mismatch tensors: " + self._format_records(adapted))
        if skipped:
            self._emit(logger, "Skipped shape-mismatch tensors: " + self._format_records(skipped))
        if missing:
            self._emit(logger, "Missing tensors after filtered load: " + self._format_names(missing))
        if unexpected:
            self._emit(logger, "Unexpected tensors ignored from checkpoint: " + self._format_names(unexpected))
        self._emit(
            logger,
            "Pose-encoding tensors are shape-compatible when image size matches; "
            f"{len(pose_loaded)} conditioningprocessor tensors exist in the baseline wrapper.",
        )

    @staticmethod
    def _emit(logger, message):
        if logger is None:
            print(message)
        else:
            logger.info(message)

    @staticmethod
    def _format_records(records, limit=12):
        shown = [f"{name}: {src}->{dst}" for name, src, dst in records[:limit]]
        suffix = "" if len(records) <= limit else f" ... (+{len(records) - limit} more)"
        return "; ".join(shown) + suffix

    @staticmethod
    def _format_names(names, limit=20):
        names = list(names)
        shown = [str(name) for name in names[:limit]]
        suffix = "" if len(names) <= limit else f" ... (+{len(names) - limit} more)"
        return "; ".join(shown) + suffix
