#!/usr/bin/env bash
set -euo pipefail

DST_HOST="${DST_HOST:-10.126.126.3}"
DST_USER="${DST_USER:-nvidia}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519_jetson2}"
KEY_COMMENT="${KEY_COMMENT:-jetson2-sync@$(hostname)}"

usage() {
  cat <<'EOF'
Usage:
  bash setup-ssh-key-to-jetson2.sh [options]

Options:
  --host HOST        Remote host. Default: 10.126.126.3
  --user USER        Remote SSH user. Default: nvidia
  --port PORT        Remote SSH port. Default: 22
  --identity PATH    SSH private key path. Default: ~/.ssh/id_ed25519_jetson2
  --comment TEXT     SSH key comment. Default: jetson2-sync@<hostname>
  -h, --help         Show this help message

What it does:
  1. Creates an ed25519 SSH key if it does not exist
  2. Installs the public key to the remote host's ~/.ssh/authorized_keys
  3. Verifies passwordless SSH login

Examples:
  bash setup-ssh-key-to-jetson2.sh
  bash setup-ssh-key-to-jetson2.sh --user nvidia --host 10.126.126.3
  bash setup-ssh-key-to-jetson2.sh --identity ~/.ssh/id_ed25519_jetson2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --comment)
      KEY_COMMENT="$2"
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

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v ssh-keygen >/dev/null 2>&1; then
  echo "ssh-keygen is required but was not found in PATH." >&2
  exit 1
fi

SSH_DIR="$(dirname "$SSH_KEY_PATH")"
PUB_KEY_PATH="${SSH_KEY_PATH}.pub"
REMOTE="${DST_USER}@${DST_HOST}"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [[ ! -f "$SSH_KEY_PATH" ]]; then
  echo "[ssh-key] generating ${SSH_KEY_PATH}"
  ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "$KEY_COMMENT"
else
  echo "[ssh-key] reusing existing key: ${SSH_KEY_PATH}"
fi

if [[ ! -f "$PUB_KEY_PATH" ]]; then
  echo "Public key was not found: ${PUB_KEY_PATH}" >&2
  exit 1
fi

PUB_KEY_CONTENT="$(cat "$PUB_KEY_PATH")"
echo "[ssh-key] installing public key on ${REMOTE}:${SSH_PORT}"
echo "[ssh-key] you may be prompted once for the remote password"

ssh -p "$SSH_PORT" "$REMOTE" \
  "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$PUB_KEY_CONTENT' ~/.ssh/authorized_keys || printf '%s\n' '$PUB_KEY_CONTENT' >> ~/.ssh/authorized_keys"

echo "[ssh-key] verifying passwordless login"
ssh -i "$SSH_KEY_PATH" -p "$SSH_PORT" -o BatchMode=yes -o IdentitiesOnly=yes "$REMOTE" "echo '[ssh-key] passwordless ssh ok on' \$(hostname)"

cat <<EOF
[ssh-key] setup complete
[ssh-key] private_key=${SSH_KEY_PATH}

You can now use:
  bash /mnt/ssd/code/Qwen3_vl/sync-to-jetson2.sh --identity ${SSH_KEY_PATH}
  bash /mnt/ssd/code/Qwen3_vl/sync-miniconda-to-jetson2.sh --identity ${SSH_KEY_PATH}

Or export once in your shell:
  export SSH_KEY_PATH=${SSH_KEY_PATH}
EOF
