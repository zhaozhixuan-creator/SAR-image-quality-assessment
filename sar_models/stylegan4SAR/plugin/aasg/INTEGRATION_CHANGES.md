# AASG Integration Change Log

This document records where AASG was integrated into the baseline StyleGAN2-ADA codebase.

## 1) `training/networks.py`

- Top-level imports:
  - Added `from plugin.aasg import AASGModule, DEFAULT_AASG_CONFIG`.
- `SynthesisNetwork.__init__`:
  - Added arguments:
    - `z_dim`
    - `aasg_enabled=False`
  - Added AASG state:
    - `self.aasg_enabled`
    - `self.aasg`
    - `self.aasg_target_resolutions`
    - `self.aasg_alpha_map`
    - `self.aasg_align_layers`
  - Added per-resolution channel align layers (`1 -> channels_dict[res]`) using `Conv2dLayer`.
- `SynthesisNetwork.forward`:
  - Added runtime args:
    - `z=None`, `c=None`
    - `aasg_enabled=None`
    - `return_aasg_viz=False`
  - Added AASG forward call:
    - `self.aasg(z=z, c=c, target_resolutions=..., return_viz=...)`
  - Added feature injection in middle/low layers:
    - `x = x + alpha_res * align(prior_res)`
  - Added optional return of AASG visualization dict.
- `Generator.__init__`:
  - Updated construction:
    - `SynthesisNetwork(z_dim=z_dim, ...)`.
- `Generator.forward`:
  - Added `return_aasg_viz=False`.
  - Passes `z,c` into synthesis.
  - Supports returning `(img, viz)` when requested.

## 2) `training/loss.py`

- `StyleGAN2Loss.run_G`:
  - Updated synthesis call:
    - from `self.G_synthesis(ws)`
    - to `self.G_synthesis(ws, z=z, c=c)`
  - Reason: AASG needs `z` and full label `c` during training.

## 3) `train.py`

- `setup_training_loop_kwargs(...)`:
  - Added argument:
    - `aasg_enabled=None`
  - Added validation:
    - `--aasg_enabled=True` requires `--cond=True`.
  - Added synthesis kwargs wiring:
    - `args.G_kwargs.synthesis_kwargs.aasg_enabled = aasg_enabled`
- CLI options:
  - Added:
    - `@click.option('--aasg_enabled', type=bool, ...)`

## 4) `generate.py`

- CLI options:
  - Added:
    - `--aasg_enabled`
- Inference path:
  - `G(..., aasg_enabled=aasg_enabled)` for seed-based generation.
  - `G.synthesis(..., aasg_enabled=aasg_enabled)` for projected-W generation.

## 5) `training/training_loop.py`

- Added helper:
  - `normalize_maps_for_viz(img)` for per-sample normalization to `[0,1]`.
- Snapshot export (initial + periodic):
  - When AASG is enabled, requests:
    - `G_ema(..., return_aasg_viz=True)`
  - Saves:
    - `fakes_init_aasg_S0.png`
    - `fakes_init_aasg_Stheta.png`
    - `fakesXXXXXX_aasg_S0.png`
    - `fakesXXXXXX_aasg_Stheta.png`

## 6) `plugin/__init__.py`

- Added plugin namespace marker to make imports explicit and stable.

---

## Notes

- When `aasg_enabled=False`, generator path remains baseline-equivalent (AASG branch is not used).
- AASG outputs are single-channel priors; channel alignment is performed in StyleGAN synthesis layers.

