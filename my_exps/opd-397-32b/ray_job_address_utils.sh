#!/usr/bin/env bash

strip_ansi_escape_codes() {
  sed -E $'s/\x1B\\[[0-9;]*[[:alpha:]]//g'
}

resolve_ray_job_address() {
  if [[ -n "${RAY_JOB_ADDRESS:-}" ]]; then
    printf '%s\n' "${RAY_JOB_ADDRESS%/}"
    return 0
  fi

  local detected_addr=""
  local job_list_output=""
  if job_list_output="$(ray job list 2>&1 | strip_ansi_escape_codes)"; then
    detected_addr="$(printf '%s\n' "${job_list_output}" | grep -Eo 'https?://[^[:space:]]+' | head -n1 || true)"
    if [[ -n "${detected_addr}" ]]; then
      printf '%s\n' "${detected_addr%/}"
      return 0
    fi
  fi

  # Last resort fallback for local clusters when auto-detection cannot parse output.
  printf 'http://%s:%s\n' "${RAY_DASHBOARD_HOST:-127.0.0.1}" "${RAY_DASHBOARD_PORT:-8265}"
}

require_ray_job_address() {
  local addr=""
  addr="$(resolve_ray_job_address)" || return 1
  addr="${addr%/}"

  if ! ray job list --address="${addr}" >/dev/null 2>&1; then
    echo "Unable to reach Ray Job server at ${addr}." >&2
    echo "Set RAY_JOB_ADDRESS explicitly, e.g. RAY_JOB_ADDRESS=http://<head-ip>:8265" >&2
    return 1
  fi

  printf '%s\n' "${addr}"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  resolve_ray_job_address
fi
