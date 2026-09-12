#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 1. Define your experiment settings here
NUM_GPUS=1
CONFIG="configs/pix2pix_SAR/pix2pix_vanilla_unet_bn_x2ka_b1x1_220k.py"
WORK_DIR="./work_dirs/x2ka_aircas_05211047"

# 2. Execute the training command
bash tools/dist_train.sh \
    "$CONFIG" \
    "$NUM_GPUS" \
    --work-dir "$WORK_DIR"
