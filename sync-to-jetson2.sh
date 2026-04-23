#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/mnt/ssd/code/Qwen3_vl}"
DST_HOST="${DST_HOST:-10.126.126.3}"
DST_USER="${DST_USER:-nvidia}"
DST_DIR="${DST_DIR:-/mnt/ssd/code/Qwen3_vl}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519_jetson2}"

DELETE_REMOTE=0
DRY_RUN=0
GIT_CHANGED=0
SYNC_PATHS=()

usage() {
  cat <<'EOF'
Usage:
  bash sync-to-jetson2.sh [options]

Options:
  --src PATH         Local source directory. Default: /mnt/ssd/code/Qwen3_vl
  --dst PATH         Remote destination directory. Default: /mnt/ssd/code/Qwen3_vl
  --host HOST        Remote host. Default: 10.126.126.3
  --user USER        Remote SSH user. Default: nvidia
  --port PORT        Remote SSH port. Default: 22
  --identity PATH    SSH private key path. Default: ~/.ssh/id_ed25519_jetson2 if it exists
  --delete           Delete remote files that no longer exist locally
  --dry-run          Show what would be synced without changing the remote side
  --git-changed      Sync only git-modified/untracked files under SRC_DIR
  --path RELPATH     Sync only a specific relative path under SRC_DIR; can be repeated
  -h, --help         Show this help message

Examples:
  bash sync-to-jetson2.sh
  bash sync-to-jetson2.sh --dry-run
  bash sync-to-jetson2.sh --delete
  bash sync-to-jetson2.sh --git-changed
  bash sync-to-jetson2.sh --path qwen3vl_tp_runtime/scripts/runtime.py
  bash sync-to-jetson2.sh --identity ~/.ssh/id_ed25519_jetson2
  DST_USER=myuser bash sync-to-jetson2.sh
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
    --identity)
      SSH_KEY_PATH="$2"
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
    --git-changed)
      GIT_CHANGED=1
      shift
      ;;
    --path)
      SYNC_PATHS+=("$2")
      shift 2
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

if [[ -n "${SSH_KEY_PATH}" && ! -f "${SSH_KEY_PATH}" ]]; then
  if [[ "${SSH_KEY_PATH}" == "$HOME/.ssh/id_ed25519_jetson2" ]]; then
    SSH_KEY_PATH=""
  else
    echo "SSH private key does not exist: ${SSH_KEY_PATH}" >&2
    exit 1
  fi
fi

REMOTE="${DST_USER}@${DST_HOST}"
RSYNC_SSH="ssh -p ${SSH_PORT}"
if [[ -n "${SSH_KEY_PATH}" ]]; then
  RSYNC_SSH+=" -i ${SSH_KEY_PATH} -o IdentitiesOnly=yes"
fi
RSYNC_CMD=(
  rsync
  -az
  --human-readable
  --exclude=__pycache__/
  --exclude=*.pyc
  --exclude=.pytest_cache/
  --exclude=.tmp_*/
  --rsync-path
  "mkdir -p \"${DST_DIR}\" && rsync"
  -e
  "${RSYNC_SSH}"
)

if [[ $DELETE_REMOTE -eq 1 ]]; then
  RSYNC_CMD+=(--delete)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_CMD+=(--dry-run --itemize-changes --stats)
else
  RSYNC_CMD+=(--info=progress2)
fi

echo "[sync] src=${SRC_DIR}"
echo "[sync] dst=${REMOTE}:${DST_DIR}"
echo "[sync] ssh_port=${SSH_PORT} delete=${DELETE_REMOTE} dry_run=${DRY_RUN} git_changed=${GIT_CHANGED}"
if [[ -n "${SSH_KEY_PATH}" ]]; then
  echo "[sync] ssh_key=${SSH_KEY_PATH}"
fi

TMP_FILE=""
cleanup() {
  if [[ -n "${TMP_FILE}" && -f "${TMP_FILE}" ]]; then
    rm -f "${TMP_FILE}"
  fi
}
trap cleanup EXIT

if [[ $GIT_CHANGED -eq 1 || ${#SYNC_PATHS[@]} -gt 0 ]]; then
  TMP_FILE="$(mktemp)"

  if [[ $GIT_CHANGED -eq 1 ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "git is required for --git-changed but was not found in PATH." >&2
      exit 1
    fi
    if ! git -C "$SRC_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
      echo "--git-changed requires SRC_DIR to be inside a git repository: $SRC_DIR" >&2
      exit 1
    fi

    while IFS= read -r path; do
      [[ -n "$path" ]] && printf '%s\n' "$path" >>"$TMP_FILE"
    done < <(git -C "$SRC_DIR" diff --name-only HEAD --)

    while IFS= read -r path; do
      [[ -n "$path" ]] && printf '%s\n' "$path" >>"$TMP_FILE"
    done < <(git -C "$SRC_DIR" ls-files --others --exclude-standard)
  fi

  if [[ ${#SYNC_PATHS[@]} -gt 0 ]]; then
    for path in "${SYNC_PATHS[@]}"; do
      printf '%s\n' "$path" >>"$TMP_FILE"
    done
  fi

  sort -u "$TMP_FILE" -o "$TMP_FILE"
  if [[ ! -s "$TMP_FILE" ]]; then
    echo "[sync] no matching files to sync"
    exit 0
  fi

  echo "[sync] syncing selected paths:"
  sed 's/^/  - /' "$TMP_FILE"
  "${RSYNC_CMD[@]}" --files-from="$TMP_FILE" "${SRC_DIR}/" "${REMOTE}:${DST_DIR}/"
else
  "${RSYNC_CMD[@]}" "${SRC_DIR}/" "${REMOTE}:${DST_DIR}/"
fi

echo "[sync] done"
