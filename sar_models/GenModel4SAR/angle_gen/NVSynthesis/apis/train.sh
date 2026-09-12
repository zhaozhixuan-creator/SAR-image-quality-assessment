#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting training process..."

python main.py \
    --mode train \
    -opt /workspace/jk_angle/configs/_base_/NVSynthesis/train.yml \
    "$@"