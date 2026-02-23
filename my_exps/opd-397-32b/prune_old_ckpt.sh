#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prune_old_ckpt.sh --ckpt-root PATH [--keep N] [--interval-sec N] [--oneshot]

Options:
  --ckpt-root PATH    Checkpoint directory containing iter_XXXXXXX subdirs (required)
  --keep N            Number of latest checkpoints to keep (default: 5)
  --interval-sec N    Prune interval in seconds for daemon mode (default: 300)
  --oneshot           Run pruning once and exit
EOF
}

CKPT_ROOT=""
KEEP_N=5
INTERVAL_SEC=300
ONE_SHOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-root)
      CKPT_ROOT="${2:-}"
      shift 2
      ;;
    --keep)
      KEEP_N="${2:-}"
      shift 2
      ;;
    --interval-sec)
      INTERVAL_SEC="${2:-}"
      shift 2
      ;;
    --oneshot)
      ONE_SHOT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "${CKPT_ROOT}" ]] || {
  echo "--ckpt-root is required" >&2
  usage >&2
  exit 1
}

[[ "${KEEP_N}" =~ ^[0-9]+$ ]] || {
  echo "--keep must be a non-negative integer, got '${KEEP_N}'" >&2
  exit 1
}

[[ "${INTERVAL_SEC}" =~ ^[0-9]+$ ]] || {
  echo "--interval-sec must be a positive integer, got '${INTERVAL_SEC}'" >&2
  exit 1
}
(( INTERVAL_SEC > 0 )) || {
  echo "--interval-sec must be > 0, got '${INTERVAL_SEC}'" >&2
  exit 1
}

prune_once() {
  [[ -d "${CKPT_ROOT}" ]] || return 0
  (( KEEP_N > 0 )) || return 0

  local ckpts=()
  mapfile -t ckpts < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' | sort)
  local count="${#ckpts[@]}"
  (( count > KEEP_N )) || return 0

  local delete_n=$((count - KEEP_N))
  for ((i = 0; i < delete_n; i++)); do
    rm -rf -- "${ckpts[$i]}"
  done
}

if (( ONE_SHOT == 1 )); then
  prune_once
  exit 0
fi

while true; do
  prune_once || true
  sleep "${INTERVAL_SEC}"
done
