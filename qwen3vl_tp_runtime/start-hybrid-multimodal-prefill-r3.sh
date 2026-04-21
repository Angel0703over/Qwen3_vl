#!/usr/bin/env bash
# Launch multimodal prefill hybrid smoke-test rank 3.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RANK=3
exec bash "${SCRIPT_DIR}/run-hybrid-multimodal-prefill-rank.sh" "$@"
