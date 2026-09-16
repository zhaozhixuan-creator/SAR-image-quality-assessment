#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/utils/translation_eval.py \
    configs/pix2pix_SAR/pix2pix_vanilla_unet_bn_x2ka_b1x1_220k.py \
    work_dirs/x2ka_aircas_pretrained/ckpt/x2ka_aircas_pretrained/latest.pth \
    --samples-path work_dirs/eval/whole_img \
    --eval none \
    --generate_whole \
    --batch_size 99 \
    --channel_list 1 2 0 \
    --n_edge 32 \
    --field all \
    --checkpoint_name W1
