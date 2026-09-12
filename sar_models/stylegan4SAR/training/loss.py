# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from plugin.ENL_bg import ENLBGModule, DEFAULT_ENLBG_CONFIG, build_enlbg_config

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cur_nimg=0): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 aasg_energy_weight=0.02, aasg_energy_tau=0.005, aasg_energy_topk=4, aasg_energy_stop_kimg=1500.0,
                 aasg_uv_center_weight=0.005, enl_bg_enabled=False):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.aasg_energy_weight = aasg_energy_weight
        self.aasg_energy_tau = aasg_energy_tau
        self.aasg_energy_topk = int(aasg_energy_topk)
        self.aasg_energy_stop_kimg = float(aasg_energy_stop_kimg)
        self.aasg_uv_center_weight = float(aasg_uv_center_weight)
        self.enl_bg_enabled = bool(enl_bg_enabled)
        self.enl_bg_module = None
        if self.enl_bg_enabled:
            cfg = build_enlbg_config(dict(DEFAULT_ENLBG_CONFIG.to_dict(), enabled=True))
            self.enl_bg_module = ENLBGModule(cfg=cfg)

    def run_G(self, z, c, sync, return_aasg_viz=False, cur_nimg=0):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            out = self.G_synthesis(ws, z=z, c=c, return_aasg_viz=return_aasg_viz, cur_nimg=cur_nimg)
        if return_aasg_viz:
            img, viz = out
            return img, ws, viz
        img = out
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cur_nimg=0):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, aasg_viz = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), return_aasg_viz=True, cur_nimg=cur_nimg) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                # Prevent AASG sparse prior collapse to pure-black maps.
                loss_aasg_energy = 0
                if isinstance(aasg_viz, dict) and 'amp' in aasg_viz and self.aasg_energy_weight > 0:
                    amp = aasg_viz['amp'].squeeze(-1)  # [N, K]
                    if amp.ndim == 2 and amp.shape[1] > 0:
                        k = min(self.aasg_energy_topk, amp.shape[1])
                        amp_topk = torch.topk(amp, k=k, dim=1).values
                        amp_topk_mean = amp_topk.mean(dim=1)
                        loss_aasg_energy = torch.relu(self.aasg_energy_tau - amp_topk_mean) * self.aasg_energy_weight
                        # Regularization decay: focus on early/mid training, then fade out.
                        if self.aasg_energy_stop_kimg > 0:
                            decay = max(0.0, 1.0 - float(cur_nimg) / (self.aasg_energy_stop_kimg * 1000.0))
                            loss_aasg_energy = loss_aasg_energy * decay
                    training_stats.report('Loss/G/aasg_energy', loss_aasg_energy)
                    loss_Gmain = loss_Gmain + loss_aasg_energy

                # Encourage sparse points to stay near the center region.
                loss_aasg_uv = 0
                if isinstance(aasg_viz, dict) and 'uv' in aasg_viz and self.aasg_uv_center_weight > 0:
                    uv = aasg_viz['uv']  # [N, K, 2], in [-1, 1]
                    if uv.ndim == 3 and uv.shape[2] == 2:
                        uv_radius = torch.sqrt((uv ** 2).sum(dim=2).clamp(min=1e-12))  # [N, K]
                        # 中心吸引项：整体压小半径。
                        uv_l2 = (uv ** 2).sum(dim=2).mean(dim=1)
                        # 边缘屏障项：半径超过阈值后惩罚快速增大，抑制向角落漂移。
                        radius_target = 0.45
                        uv_barrier = torch.relu(uv_radius - radius_target).pow(2).mean(dim=1)
                        loss_aasg_uv = uv_l2 * self.aasg_uv_center_weight + uv_barrier * (self.aasg_uv_center_weight * 4.0)
                        training_stats.report('Loss/G/aasg_uv_center', loss_aasg_uv)
                        training_stats.report('Loss/G/aasg_uv_radius_mean', uv_radius.mean(dim=1))
                        loss_Gmain = loss_Gmain + loss_aasg_uv

                # ENL_bg: background statistical consistency regularization.
                if self.enl_bg_module is not None:
                    enl_bg_loss, enl_bg_stats = self.enl_bg_module(
                        gen_img=gen_img,
                        real_img=real_img,
                        cur_nimg=cur_nimg,
                        return_stats=True,
                    )
                    loss_Gmain = loss_Gmain + enl_bg_loss
                    training_stats.report('Loss/G/enl_bg', enl_bg_loss)
                    training_stats.report('Loss/ENL/enl_gen_mean', torch.as_tensor(enl_bg_stats['enl_gen_mean'], device=self.device))
                    training_stats.report('Loss/ENL/enl_real_mean', torch.as_tensor(enl_bg_stats['enl_real_mean'], device=self.device))
                    training_stats.report('Loss/ENL/lambda_eff', torch.as_tensor(enl_bg_stats['lambda_eff'], device=self.device))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync, cur_nimg=cur_nimg)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False, cur_nimg=cur_nimg)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
