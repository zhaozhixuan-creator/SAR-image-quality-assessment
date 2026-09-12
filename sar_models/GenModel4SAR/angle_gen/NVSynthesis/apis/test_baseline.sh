#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

echo "Starting baseline testing process..."

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" main.py \
    --mode test \
    -opt "${REPO_ROOT}/configs/_base_/NVSynthesis/baseline_test.yml" \
    "$@"
