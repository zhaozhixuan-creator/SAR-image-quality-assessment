# ENL_bg Integration Change Log

This file documents the integration changes required to enable module-3 (ENL_bg) in training.

## 1) `training/loss.py`

- Added imports:
  - `ENLBGModule`
  - `DEFAULT_ENLBG_CONFIG`
  - `build_enlbg_config`
- Extended `StyleGAN2Loss.__init__` with:
  - `enl_bg_enabled=False`
- If enabled:
  - Build config from plugin defaults (`enabled=True`)
  - Instantiate `self.enl_bg_module`
- In `Gmain` branch:
  - Compute ENL_bg loss:
    - `enl_bg_loss, enl_bg_stats = self.enl_bg_module(gen_img, real_img, cur_nimg, return_stats=True)`
  - Add to generator objective:
    - `loss_Gmain += enl_bg_loss`
  - Report training stats:
    - `Loss/G/enl_bg`
    - `Loss/ENL/enl_gen_mean`
    - `Loss/ENL/enl_real_mean`
    - `Loss/ENL/lambda_eff`

## 2) `train.py`

- Added setup argument:
  - `enl_bg_enabled=None`
- Added CLI option:
  - `--enl_bg_enabled`
- Wired to loss kwargs:
  - `args.loss_kwargs.enl_bg_enabled = enl_bg_enabled`
- Kept all ENL_bg hyperparameters centralized in:
  - `plugin/ENL_bg/ENL_bgconfig.py`

## 3) Not changed intentionally

- `generate.py`: no ENL_bg switch added.
- `compare_batch.py`: no ENL_bg switch added.

Reason: ENL_bg is a training-time loss term only and does not change inference graph directly.

