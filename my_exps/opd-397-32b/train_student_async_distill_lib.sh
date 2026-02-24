#!/bin/bash

DISTILL_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RAY_ADDR_UTIL="${DISTILL_LIB_DIR}/ray_job_address_utils.sh"
[[ -f "${RAY_ADDR_UTIL}" ]] || {
  echo "Missing Ray address helper script: ${RAY_ADDR_UTIL}" >&2
  exit 1
}
# shellcheck source=/dev/null
source "${RAY_ADDR_UTIL}"

require_file() {
  local path="$1"
  local message="$2"
  [[ -f "${path}" ]] || {
    echo "${message}: ${path}" >&2
    exit 1
  }
}

require_non_negative_int() {
  local value="$1"
  local name="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "${name} must be a non-negative integer, got '${value}'" >&2
    exit 1
  }
}

require_positive_int() {
  local value="$1"
  local name="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "${name} must be a positive integer, got '${value}'" >&2
    exit 1
  }
  (( value > 0 )) || {
    echo "${name} must be > 0, got '${value}'" >&2
    exit 1
  }
}

require_float_range() {
  local value="$1"
  local name="$2"
  local min="$3"
  local max="$4"

  if ! python3 - "${value}" "${min}" "${max}" <<'PY'
import sys

try:
    x = float(sys.argv[1])
    lo = float(sys.argv[2])
    hi = float(sys.argv[3])
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if lo <= x <= hi else 1)
PY
  then
    echo "${name} must be a float in [${min},${max}], got '${value}'" >&2
    exit 1
  fi
}

normalize_bool_flag() {
  local value="$1"
  local name="$2"
  value="$(echo "${value}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|y|on)
      echo "1"
      ;;
    0|false|no|n|off|"")
      echo "0"
      ;;
    *)
      echo "${name} must be one of: 0/1, true/false, yes/no, on/off. Got '${value}'" >&2
      exit 1
      ;;
  esac
}

yaml_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "${value}"
}

gcd() {
  local a="$1"
  local b="$2"
  while (( b != 0 )); do
    local t="$b"
    b=$((a % b))
    a="$t"
  done
  echo "$a"
}

detect_rollout_context_len() {
  local hf_path="$1"
  local config_path="${hf_path%/}/config.json"
  [[ -f "${config_path}" ]] || return 1

  local context_len=""
  context_len="$(awk -F: '/"max_position_embeddings"[[:space:]]*:/ {gsub(/[^0-9]/, "", $2); if (length($2) > 0) {print $2; exit}}' "${config_path}")"
  if [[ -z "${context_len}" ]]; then
    context_len="$(awk -F: '/"model_max_length"[[:space:]]*:/ {gsub(/[^0-9]/, "", $2); if (length($2) > 0) {print $2; exit}}' "${config_path}")"
  fi

  [[ -n "${context_len}" ]] || return 1
  echo "${context_len}"
}

validate_rollout_lengths() {
  for len_var in ROLLOUT_MAX_CONTEXT_LEN ROLLOUT_MAX_PROMPT_LEN ROLLOUT_MAX_RESPONSE_LEN; do
    require_positive_int "${!len_var}" "${len_var}"
  done

  (( ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN <= ROLLOUT_MAX_CONTEXT_LEN )) || {
    echo "Invalid rollout length limits:" >&2
    echo "  rollout_max_prompt_len(${ROLLOUT_MAX_PROMPT_LEN}) + rollout_max_response_len(${ROLLOUT_MAX_RESPONSE_LEN})" >&2
    echo "  exceeds rollout_max_context_len(${ROLLOUT_MAX_CONTEXT_LEN})" >&2
    exit 1
  }
}

setup_distill_mode() {
  DISTILL_LOSS_MODE="${DISTILL_LOSS_MODE:-rkl}"
  DISTILL_LOSS_MODE="$(echo "${DISTILL_LOSS_MODE}" | tr '[:upper:]' '[:lower:]')"
  case "${DISTILL_LOSS_MODE}" in
    rkl|fkl|mixed|jsd) ;;
    *)
      echo "DISTILL_LOSS_MODE must be one of: rkl, fkl, mixed, jsd. Got '${DISTILL_LOSS_MODE}'" >&2
      exit 1
      ;;
  esac

  OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
  require_float_range "${OPD_KL_COEF}" "OPD_KL_COEF" "0" "1000000"

  OPD_TOP_LOGPROBS_NUM="${OPD_TOP_LOGPROBS_NUM:-16}"
  require_positive_int "${OPD_TOP_LOGPROBS_NUM}" "OPD_TOP_LOGPROBS_NUM"

  OPD_MIXED_KL_WEIGHT="${OPD_MIXED_KL_WEIGHT:-0.5}"
  require_float_range "${OPD_MIXED_KL_WEIGHT}" "OPD_MIXED_KL_WEIGHT" "0" "1"

  OPD_DISTILL_COEF="${OPD_DISTILL_COEF:-1.0}"
  require_float_range "${OPD_DISTILL_COEF}" "OPD_DISTILL_COEF" "0" "1000000"

  OPD_JSD_BETA="${OPD_JSD_BETA:-0.5}"
  require_float_range "${OPD_JSD_BETA}" "OPD_JSD_BETA" "0" "1"

  OPD_PRIVILEGED_ENABLE="$(normalize_bool_flag "${OPD_PRIVILEGED_ENABLE:-0}" "OPD_PRIVILEGED_ENABLE")"
  OPD_PRIVILEGED_METADATA_KEY="${OPD_PRIVILEGED_METADATA_KEY:-privileged_context}"
  OPD_PRIVILEGED_FALLBACK_LABEL="$(normalize_bool_flag "${OPD_PRIVILEGED_FALLBACK_LABEL:-1}" "OPD_PRIVILEGED_FALLBACK_LABEL")"
  OPD_PRIVILEGED_TOKENIZER_PATH="${OPD_PRIVILEGED_TOKENIZER_PATH:-}"

  OPD_RM_CONNECT_TIMEOUT_S="${OPD_RM_CONNECT_TIMEOUT_S:-2.0}"
  OPD_RM_READ_TIMEOUT_S="${OPD_RM_READ_TIMEOUT_S:-120.0}"
  OPD_RM_TOTAL_TIMEOUT_S="${OPD_RM_TOTAL_TIMEOUT_S:-180.0}"
  OPD_RM_MAX_CONNECTIONS="${OPD_RM_MAX_CONNECTIONS:-512}"
  OPD_RM_MAX_CONNECTIONS_PER_HOST="${OPD_RM_MAX_CONNECTIONS_PER_HOST:-256}"
  OPD_RM_RETRY_ATTEMPTS="${OPD_RM_RETRY_ATTEMPTS:-6}"
  OPD_RM_RETRY_BASE_SLEEP_S="${OPD_RM_RETRY_BASE_SLEEP_S:-0.15}"
  OPD_RM_RETRY_MAX_SLEEP_S="${OPD_RM_RETRY_MAX_SLEEP_S:-2.0}"
  require_float_range "${OPD_RM_CONNECT_TIMEOUT_S}" "OPD_RM_CONNECT_TIMEOUT_S" "0.001" "3600"
  require_float_range "${OPD_RM_READ_TIMEOUT_S}" "OPD_RM_READ_TIMEOUT_S" "0.001" "7200"
  require_float_range "${OPD_RM_TOTAL_TIMEOUT_S}" "OPD_RM_TOTAL_TIMEOUT_S" "0.001" "7200"
  require_positive_int "${OPD_RM_MAX_CONNECTIONS}" "OPD_RM_MAX_CONNECTIONS"
  require_positive_int "${OPD_RM_MAX_CONNECTIONS_PER_HOST}" "OPD_RM_MAX_CONNECTIONS_PER_HOST"
  require_positive_int "${OPD_RM_RETRY_ATTEMPTS}" "OPD_RM_RETRY_ATTEMPTS"
  require_float_range "${OPD_RM_RETRY_BASE_SLEEP_S}" "OPD_RM_RETRY_BASE_SLEEP_S" "0.001" "60"
  require_float_range "${OPD_RM_RETRY_MAX_SLEEP_S}" "OPD_RM_RETRY_MAX_SLEEP_S" "0.001" "120"

  OPD_DEBUG_DUMP_ENABLE="$(normalize_bool_flag "${OPD_DEBUG_DUMP_ENABLE:-1}" "OPD_DEBUG_DUMP_ENABLE")"
  OPD_DEBUG_DUMP_DIR="${OPD_DEBUG_DUMP_DIR:-}"
  OPD_DEBUG_DUMP_MAX_TOTAL_MB="${OPD_DEBUG_DUMP_MAX_TOTAL_MB:-5120}"
  OPD_DEBUG_DUMP_MAX_FILES="${OPD_DEBUG_DUMP_MAX_FILES:-20000}"
  OPD_DEBUG_DUMP_MAX_FILE_MB="${OPD_DEBUG_DUMP_MAX_FILE_MB:-64}"
  OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE="${OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE:-8}"
  OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE="${OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE:-512}"
  OPD_DEBUG_DUMP_SEED="${OPD_DEBUG_DUMP_SEED:-${SEED:-1234}}"
  require_non_negative_int "${OPD_DEBUG_DUMP_MAX_TOTAL_MB}" "OPD_DEBUG_DUMP_MAX_TOTAL_MB"
  require_positive_int "${OPD_DEBUG_DUMP_MAX_FILES}" "OPD_DEBUG_DUMP_MAX_FILES"
  require_positive_int "${OPD_DEBUG_DUMP_MAX_FILE_MB}" "OPD_DEBUG_DUMP_MAX_FILE_MB"
  require_positive_int "${OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE}" "OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE"
  require_positive_int "${OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE}" "OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE"
  require_non_negative_int "${OPD_DEBUG_DUMP_SEED}" "OPD_DEBUG_DUMP_SEED"
}

build_distill_args() {
  DISTILL_CUSTOM_CONFIG_PATH=""
  DISTILL_ARGS=()

  DISTILL_CUSTOM_CONFIG_PATH="${SCRIPT_DIR}/.distill_config_${DISTILL_LOSS_MODE}.yaml"
  cat > "${DISTILL_CUSTOM_CONFIG_PATH}" <<EOF
distill_loss_mode: ${DISTILL_LOSS_MODE}
opd_debug_dump_enable: ${OPD_DEBUG_DUMP_ENABLE}
opd_debug_dump_dir: "$(yaml_escape "${OPD_DEBUG_DUMP_DIR}")"
opd_debug_dump_max_total_mb: ${OPD_DEBUG_DUMP_MAX_TOTAL_MB}
opd_debug_dump_max_files: ${OPD_DEBUG_DUMP_MAX_FILES}
opd_debug_dump_max_file_mb: ${OPD_DEBUG_DUMP_MAX_FILE_MB}
opd_debug_dump_max_samples_per_update: ${OPD_DEBUG_DUMP_MAX_SAMPLES_PER_UPDATE}
opd_debug_dump_max_positions_per_sample: ${OPD_DEBUG_DUMP_MAX_POSITIONS_PER_SAMPLE}
opd_debug_dump_seed: ${OPD_DEBUG_DUMP_SEED}
EOF

  if [[ "${DISTILL_LOSS_MODE}" == "rkl" ]]; then
    DISTILL_ARGS+=(
      --use-opd
      --opd-type sglang
      --opd-kl-coef "${OPD_KL_COEF}"
      --custom-config-path "${DISTILL_CUSTOM_CONFIG_PATH}"
      --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
      --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
    )
    return
  fi

  [[ "${CONTEXT_PARALLEL_SIZE}" == "1" ]] || {
    echo "DISTILL_LOSS_MODE='${DISTILL_LOSS_MODE}' currently requires CONTEXT_PARALLEL_SIZE=1." >&2
    exit 1
  }

  cat >> "${DISTILL_CUSTOM_CONFIG_PATH}" <<EOF
opd_top_logprobs_num: ${OPD_TOP_LOGPROBS_NUM}
opd_mixed_kl_weight: ${OPD_MIXED_KL_WEIGHT}
opd_distill_coef: ${OPD_DISTILL_COEF}
opd_privileged_enable: ${OPD_PRIVILEGED_ENABLE}
opd_privileged_metadata_key: "$(yaml_escape "${OPD_PRIVILEGED_METADATA_KEY}")"
opd_privileged_fallback_label: ${OPD_PRIVILEGED_FALLBACK_LABEL}
opd_privileged_tokenizer_path: "$(yaml_escape "${OPD_PRIVILEGED_TOKENIZER_PATH}")"
opd_rm_connect_timeout_s: ${OPD_RM_CONNECT_TIMEOUT_S}
opd_rm_read_timeout_s: ${OPD_RM_READ_TIMEOUT_S}
opd_rm_total_timeout_s: ${OPD_RM_TOTAL_TIMEOUT_S}
opd_rm_max_connections: ${OPD_RM_MAX_CONNECTIONS}
opd_rm_max_connections_per_host: ${OPD_RM_MAX_CONNECTIONS_PER_HOST}
opd_rm_retry_attempts: ${OPD_RM_RETRY_ATTEMPTS}
opd_rm_retry_base_sleep_s: ${OPD_RM_RETRY_BASE_SLEEP_S}
opd_rm_retry_max_sleep_s: ${OPD_RM_RETRY_MAX_SLEEP_S}
EOF

  if [[ "${DISTILL_LOSS_MODE}" == "jsd" ]]; then
    echo "opd_jsd_beta: ${OPD_JSD_BETA}" >> "${DISTILL_CUSTOM_CONFIG_PATH}"
  fi

  DISTILL_ARGS+=(
    --loss-type custom_loss
    --custom-loss-function-path opd_topk_loss_plugin.distill_topk_custom_loss
    --disable-compute-advantages-and-returns
    --custom-config-path "${DISTILL_CUSTOM_CONFIG_PATH}"
    --custom-rm-path opd_topk_reward_plugin.reward_func_topk
    --custom-reward-post-process-path opd_topk_reward_plugin.post_process_rewards_topk
  )
}

cleanup_distill_temp_file() {
  if [[ -n "${DISTILL_CUSTOM_CONFIG_PATH:-}" && -f "${DISTILL_CUSTOM_CONFIG_PATH}" ]]; then
    rm -f "${DISTILL_CUSTOM_CONFIG_PATH}" || true
  fi
}
