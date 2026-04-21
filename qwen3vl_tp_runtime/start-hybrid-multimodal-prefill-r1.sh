#!/usr/bin/env bash
# Launch multimodal prefill hybrid smoke-test rank 1.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RANK=1
exec bash "${SCRIPT_DIR}/run-hybrid-multimodal-prefill-rank.sh" "$@"
