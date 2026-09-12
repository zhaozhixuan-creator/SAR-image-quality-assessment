#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1

echo "Starting testing process..."
echo "Default target filter: ZIL131. Use '--test-target-type all' to evaluate all test targets."

"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
    --mode test \
    -opt "${REPO_ROOT}/configs/_base_/NVSynthesis/test.yml" \
    "$@"
