#!/usr/bin/env bash
# Run the multimodal decode hybrid regression checker in one command.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ssd/miniconda3/envs/vlm/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/checks/check_hybrid_multimodal_decode.py" "$@"
