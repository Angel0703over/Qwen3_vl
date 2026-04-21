#!/usr/bin/env bash
# Launch multimodal decode hybrid smoke-test rank 0.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RANK=0
exec bash "${SCRIPT_DIR}/run-hybrid-multimodal-decode-rank.sh" "$@"
