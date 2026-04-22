#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/mnt/ssd/miniconda3}"
DST_HOST="${DST_HOST:-10.126.126.3}"
DST_USER="${DST_USER:-nvidia}"
DST_DIR="${DST_DIR:-/mnt/ssd/miniconda3}"
SSH_PORT="${SSH_PORT:-22}"

DELETE_REMOTE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash sync-miniconda-to-jetson2.sh [options]

Options:
  --src PATH         Local source directory. Default: /mnt/ssd/miniconda3
  --dst PATH         Remote destination directory. Default: /mnt/ssd/miniconda3
  --host HOST        Remote host. Default: 10.126.126.3
  --user USER        Remote SSH user. Default: nvidia
  --port PORT        Remote SSH port. Default: 22
  --delete           Delete remote files that no longer exist locally
  --dry-run          Show what would be synced without changing the remote side
  -h, --help         Show this help message

Examples:
  bash sync-miniconda-to-jetson2.sh
  bash sync-miniconda-to-jetson2.sh --dry-run
  bash sync-miniconda-to-jetson2.sh --delete
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_DIR="$2"
      shift 2
      ;;
    --dst)
      DST_DIR="$2"
      shift 2
      ;;
    --host)
      DST_HOST="$2"
      shift 2
      ;;
    --user)
      DST_USER="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --delete)
      DELETE_REMOTE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but was not found in PATH." >&2
  exit 1
fi

REMOTE="${DST_USER}@${DST_HOST}"
RSYNC_CMD=(
  rsync
  -az
  --human-readable
  --exclude=pkgs/cache/
  --exclude=envs/*/pkgs/
  --exclude=envs/*/.cache/
  --rsync-path
  "mkdir -p \"${DST_DIR}\" && rsync"
  -e
  "ssh -p ${SSH_PORT}"
)

if [[ $DELETE_REMOTE -eq 1 ]]; then
  RSYNC_CMD+=(--delete)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_CMD+=(--dry-run --itemize-changes --stats)
else
  RSYNC_CMD+=(--info=progress2)
fi

echo "[sync-conda] src=${SRC_DIR}"
echo "[sync-conda] dst=${REMOTE}:${DST_DIR}"
echo "[sync-conda] ssh_port=${SSH_PORT} delete=${DELETE_REMOTE} dry_run=${DRY_RUN}"

"${RSYNC_CMD[@]}" "${SRC_DIR}/" "${REMOTE}:${DST_DIR}/"

echo "[sync-conda] done"
