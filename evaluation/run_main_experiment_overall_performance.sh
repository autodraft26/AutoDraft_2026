#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_XML="${CONFIG_XML-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-xml)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --config-xml requires a path" >&2
        exit 2
      fi
      CONFIG_XML="$2"
      shift 2
      ;;
    --config-xml=*)
      CONFIG_XML="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash evaluation/run_main_experiment_overall_performance.sh [--config-xml PATH]

Configuration precedence:
  1. Environment variables supplied by the user
  2. Values in --config-xml / CONFIG_XML
  3. Script defaults

XML supports either direct tags, e.g. <MAX_NEW_TOKENS>256</MAX_NEW_TOKENS>,
or parameter entries, e.g. <parameter name="MAX_NEW_TOKENS">256</parameter>.
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      echo "Usage: bash evaluation/run_main_experiment_overall_performance.sh [--config-xml PATH]" >&2
      exit 2
      ;;
  esac
done

load_config_xml() {
  local config_path="$1"
  [[ -n "${config_path}" ]] || return 0
  if [[ ! -f "${config_path}" ]]; then
    echo "[ERROR] config XML not found: ${config_path}" >&2
    exit 2
  fi

  eval "$(
    CONFIG_XML_PATH="${config_path}" python3 - <<'PY'
import os
import re
import shlex
import sys
import xml.etree.ElementTree as ET

allowed = {
    "TARGET_HOST",
    "TARGET_PORT",
    "BASE_MODEL_PATH",
    "DRAFT_MODEL_PATH",
    "TOKENIZER_PATH",
    "DEVICE_MAP",
    "DRAFT_DEVICE_NAME",
    "DRAFT_ENTRYPOINT",
    "SERVER_NAME",
    "TARGET_QUANTIZATION",
    "DRAFT_QUANTIZATION",
    "TEMPERATURE",
    "NUM_CHOICES",
    "SEED",
    "DETERMINISTIC",
    "OBJECTIVE_METRIC",
    "OBJECTIVE_METRICS_CSV",
    "BASE_COST_SENSITIVITY",
    "AUTODRAFT_CS_LIST",
    "AUTODRAFT_ABLATION_MODES_CSV",
    "SKIP_SELECTED_EXISTING_AUTODRAFT",
    "SELECTED_RESULT_ROOT",
    "DRAFT_PER_HOUR_COST",
    "TARGET_PER_HOUR_COST",
    "DRAFT_ELECTRICITY_COST_PER_KWH",
    "USER_COMM_COST_PER_GB",
    "CLOUD_OUTBOUND_COST_PER_GB",
    "ACCEPT_LENGTH_MARGIN",
    "PROPOSED_NODES",
    "PROPOSED_MAX_DEPTH",
    "PROACTIVE_THRESHOLD",
    "JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD",
    "DISABLE_PROACTIVE_BUDGET",
    "FIXED_TREE_WIDTH",
    "FIXED_TREE_DEPTH",
    "FIXED_TREE_NODES",
    "NON_TREE_WIDTH",
    "NON_TREE_DEPTH",
    "NON_TREE_NODES",
    "BENCHES_CSV",
    "QUESTION_LIMIT",
    "ENABLE_GPU_MONITOR",
    "GPU_MONITOR_INTERVAL",
    "DISABLE_AUTO_PROFILE",
    "DISABLE_ONLINE_PROFILE_UPDATE",
    "FORCE_PROFILE_REFRESH",
    "REFERENCE_FORCE_REFRESH",
    "ENABLE_SERVER_ONLY_AR",
    "ENABLE_SERVER_ONLY_NON_TREE_SD",
    "ENABLE_SERVER_ONLY_AUTODRAFT",
    "ENABLE_HYBRID_AUTODRAFT",
    "ENABLE_HYBRID_NON_TREE_SD",
    "ENABLE_HYBRID_OPT_TREE",
    "ENABLE_SERVER_ONLY_OPT_TREE",
    "RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE",
    "SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY",
    "SERVER_DRAFT_PROFILE_AUTO",
    "SERVER_DRAFT_PROFILE_FORCE_REFRESH",
    "SERVER_ONLY_AUTODRAFT_CS_LIST",
    "TARGET_PROFILE_MODEL_CALLS_PER_COUNT",
    "TARGET_PROFILE_NODE_LIST",
    "SERVER_DRAFT_PROFILE_MODEL_CALLS_PER_COUNT",
    "SERVER_DRAFT_PROFILE_WIDTH_LIST",
    "REFERENCE_CS_CURVE_ROUNDS",
    "REFERENCE_MAX_STEPS_LIMIT",
    "MAX_NEW_TOKENS",
    "SERVER_ONLY_AR_MAX_NEW_TOKENS",
    "SERVER_ONLY_AR_TURN_RPC",
    "DRAFT_PROFILE_MODEL_CALLS_PER_COUNT",
    "DRAFT_PROFILE_WIDTH_LIST",
    "DRY_RUN",
    "SKIP_SLEEP",
    "SLEEP_SECONDS",
}

def normalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()

path = os.environ["CONFIG_XML_PATH"]
try:
    root = ET.parse(path).getroot()
except Exception as exc:
    print(f"echo {shlex.quote('[ERROR] failed to parse config XML: ' + str(exc))} >&2")
    print("exit 2")
    sys.exit(0)

values = {}
for elem in root.iter():
    text = (elem.text or "").strip()
    if not text:
        continue
    if "name" in elem.attrib:
        key = normalize(elem.attrib["name"])
    elif list(elem):
        continue
    else:
        key = normalize(elem.tag)
    if key in allowed:
        values[key] = text

for key, value in values.items():
    if key in os.environ:
        continue
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
}

load_config_xml "${CONFIG_XML}"

# -----------------------------
# Runtime target / model config
# -----------------------------
TARGET_HOST="${TARGET_HOST:-192.168.0.12}"
TARGET_PORT="${TARGET_PORT:-26001}"

# Requested pair
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL_PATH}}"

DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
DRAFT_DEVICE_NAME="${DRAFT_DEVICE_NAME:-rtx5080}"
DRAFT_ENTRYPOINT="${DRAFT_ENTRYPOINT:-evaluation.eval_autodraft_draft}"
SERVER_NAME="${SERVER_NAME:-rtxproa6000}"

TARGET_QUANTIZATION="${TARGET_QUANTIZATION:-none}" # none|4bit|8bit|auto
DRAFT_QUANTIZATION="${DRAFT_QUANTIZATION:-none}"   # none|4bit|8bit

TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_CHOICES="${NUM_CHOICES:-1}"
SEED="${SEED:-4}"
DETERMINISTIC="${DETERMINISTIC:-0}"

# -----------------------------
# Cost / objective config
# -----------------------------
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-total_cost}"
OBJECTIVE_METRICS_CSV="${OBJECTIVE_METRICS_CSV:-${OBJECTIVE_METRIC}}"
IFS=',' read -r -a OBJECTIVE_METRICS <<< "${OBJECTIVE_METRICS_CSV}"
PRIMARY_OBJECTIVE_METRIC="${OBJECTIVE_METRICS[0]}"
BASE_COST_SENSITIVITY="${BASE_COST_SENSITIVITY:-0.5}"
AUTODRAFT_CS_LIST="${AUTODRAFT_CS_LIST:-0 0.5 1}"
AUTODRAFT_ABLATION_MODES_CSV="${AUTODRAFT_ABLATION_MODES_CSV:-full}"
IFS=',' read -r -a AUTODRAFT_ABLATION_MODES <<< "${AUTODRAFT_ABLATION_MODES_CSV}"
SKIP_SELECTED_EXISTING_AUTODRAFT="${SKIP_SELECTED_EXISTING_AUTODRAFT:-0}"
SELECTED_RESULT_ROOT="${SELECTED_RESULT_ROOT:-${ROOT_DIR}/data/selected_result/main_experiment_overall_performance_final}"

DRAFT_PER_HOUR_COST="${DRAFT_PER_HOUR_COST:-0.152}"
TARGET_PER_HOUR_COST="${TARGET_PER_HOUR_COST:-1.208}"
DRAFT_ELECTRICITY_COST_PER_KWH="${DRAFT_ELECTRICITY_COST_PER_KWH:-0.2}"
if [[ ",${OBJECTIVE_METRICS_CSV}," == *",total_cost,"* ]]; then
  USER_COMM_COST_PER_GB="${USER_COMM_COST_PER_GB:-0.33}"
  CLOUD_OUTBOUND_COST_PER_GB="${CLOUD_OUTBOUND_COST_PER_GB:-0.09}"
else
  USER_COMM_COST_PER_GB="${USER_COMM_COST_PER_GB:-0}"
  CLOUD_OUTBOUND_COST_PER_GB="${CLOUD_OUTBOUND_COST_PER_GB:-0}"
fi
ACCEPT_LENGTH_MARGIN="${ACCEPT_LENGTH_MARGIN:-0.05}"

# -----------------------------
# Tree config
# -----------------------------
PROPOSED_NODES="${PROPOSED_NODES:-150}"
PROPOSED_MAX_DEPTH="${PROPOSED_MAX_DEPTH:-15}"
PROACTIVE_THRESHOLD="${PROACTIVE_THRESHOLD:-0.0}"
JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD="${JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD:-1}"
DISABLE_PROACTIVE_BUDGET="${DISABLE_PROACTIVE_BUDGET:-1}"

FIXED_TREE_WIDTH="${FIXED_TREE_WIDTH:-150}"
FIXED_TREE_DEPTH="${FIXED_TREE_DEPTH:-15}"
FIXED_TREE_NODES="${FIXED_TREE_NODES:-150}"

NON_TREE_WIDTH="${NON_TREE_WIDTH:-1}"
NON_TREE_DEPTH="${NON_TREE_DEPTH:-15}"
NON_TREE_NODES="${NON_TREE_NODES:-15}"

# -----------------------------
# Dataset / run controls
# -----------------------------
# Start with MT-bench only by default.
# Expand later with:
# BENCHES_CSV="mt_bench,gsm8k,cnn_dailymail,humaneval,ifeval,math-500"
BENCHES_CSV="${BENCHES_CSV:-mt_bench}"
IFS=',' read -r -a BENCHES <<< "${BENCHES_CSV}"

QUESTION_LIMIT="${QUESTION_LIMIT:-}"
ENABLE_GPU_MONITOR="${ENABLE_GPU_MONITOR:-1}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-0.05}"
DISABLE_AUTO_PROFILE="${DISABLE_AUTO_PROFILE:-0}"
DISABLE_ONLINE_PROFILE_UPDATE="${DISABLE_ONLINE_PROFILE_UPDATE:-0}"
FORCE_PROFILE_REFRESH="${FORCE_PROFILE_REFRESH:-1}"
REFERENCE_FORCE_REFRESH="${REFERENCE_FORCE_REFRESH:-1}"
ENABLE_SERVER_ONLY_AR="${ENABLE_SERVER_ONLY_AR:-0}"
ENABLE_SERVER_ONLY_NON_TREE_SD="${ENABLE_SERVER_ONLY_NON_TREE_SD:-0}"
ENABLE_SERVER_ONLY_AUTODRAFT="${ENABLE_SERVER_ONLY_AUTODRAFT:-1}"
ENABLE_HYBRID_AUTODRAFT="${ENABLE_HYBRID_AUTODRAFT:-1}"
ENABLE_HYBRID_NON_TREE_SD="${ENABLE_HYBRID_NON_TREE_SD:-0}"
ENABLE_HYBRID_OPT_TREE="${ENABLE_HYBRID_OPT_TREE:-0}"
ENABLE_SERVER_ONLY_OPT_TREE="${ENABLE_SERVER_ONLY_OPT_TREE:-0}"
RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE="${RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE:-1}"
SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY="${SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY:-1}"
SERVER_DRAFT_PROFILE_AUTO="${SERVER_DRAFT_PROFILE_AUTO:-1}"
SERVER_DRAFT_PROFILE_FORCE_REFRESH="${SERVER_DRAFT_PROFILE_FORCE_REFRESH:-1}"
SERVER_ONLY_AUTODRAFT_CS_LIST="${SERVER_ONLY_AUTODRAFT_CS_LIST:-${AUTODRAFT_CS_LIST}}"
TARGET_PROFILE_MODEL_CALLS_PER_COUNT="${TARGET_PROFILE_MODEL_CALLS_PER_COUNT:-10}"
TARGET_PROFILE_NODE_LIST="${TARGET_PROFILE_NODE_LIST:-10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}"
SERVER_DRAFT_PROFILE_MODEL_CALLS_PER_COUNT="${SERVER_DRAFT_PROFILE_MODEL_CALLS_PER_COUNT:-100}"
SERVER_DRAFT_PROFILE_WIDTH_LIST="${SERVER_DRAFT_PROFILE_WIDTH_LIST:-10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}"
REFERENCE_CS_CURVE_ROUNDS="${REFERENCE_CS_CURVE_ROUNDS:-20}"
REFERENCE_MAX_STEPS_LIMIT="${REFERENCE_MAX_STEPS_LIMIT:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SERVER_ONLY_AR_MAX_NEW_TOKENS="${SERVER_ONLY_AR_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
SERVER_ONLY_AR_TURN_RPC="${SERVER_ONLY_AR_TURN_RPC:-1}"
DRAFT_PROFILE_MODEL_CALLS_PER_COUNT="${DRAFT_PROFILE_MODEL_CALLS_PER_COUNT:-100}"
DRAFT_PROFILE_WIDTH_LIST="${DRAFT_PROFILE_WIDTH_LIST:-10,20,30,40,50,60,70,80,90,100,110,120,130,140,150}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_SLEEP="${SKIP_SLEEP:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/result/main_experiment_overall_performance_${RUN_TS}"
ANSWER_DIR="${RUN_ROOT}/answers"
LOG_DIR="${RUN_ROOT}/logs"
META_JSONL="${RUN_ROOT}/runs_meta.jsonl"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
mkdir -p "${ANSWER_DIR}" "${LOG_DIR}"

TOTAL_RUNS=0
SUCCESS_COUNT=0
FAIL_COUNT=0

COMMON_ARGS=(
  --host "${TARGET_HOST}"
  --port "${TARGET_PORT}"
  --base-model-path "${BASE_MODEL_PATH}"
  --draft-model-path "${DRAFT_MODEL_PATH}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --temperature "${TEMPERATURE}"
  --device-map "${DEVICE_MAP}"
  --num-choices "${NUM_CHOICES}"
  --device-name "${DRAFT_DEVICE_NAME}"
  --server-name "${SERVER_NAME}"
  --target-quantization "${TARGET_QUANTIZATION}"
  --draft-per-hour-cost "${DRAFT_PER_HOUR_COST}"
  --target-per-hour-cost "${TARGET_PER_HOUR_COST}"
  --draft-electricity-cost-per-kwh "${DRAFT_ELECTRICITY_COST_PER_KWH}"
  --user-communication-cost-per-gb "${USER_COMM_COST_PER_GB}"
  --cloud-outbound-cost-per-gb "${CLOUD_OUTBOUND_COST_PER_GB}"
  --accept-length-margin "${ACCEPT_LENGTH_MARGIN}"
  --draft-profile-model-calls-per-count "${DRAFT_PROFILE_MODEL_CALLS_PER_COUNT}"
  --draft-profile-width-list "${DRAFT_PROFILE_WIDTH_LIST}"
  --target-profile-model-calls-per-count "${TARGET_PROFILE_MODEL_CALLS_PER_COUNT}"
  --target-profile-node-list "${TARGET_PROFILE_NODE_LIST}"
  --reference-cs-curve-rounds "${REFERENCE_CS_CURVE_ROUNDS}"
  --reference-max-steps-limit "${REFERENCE_MAX_STEPS_LIMIT}"
  --gpu-monitor-interval "${GPU_MONITOR_INTERVAL}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${DRAFT_QUANTIZATION}" == "4bit" ]]; then
  COMMON_ARGS+=(--load-in-4bit)
elif [[ "${DRAFT_QUANTIZATION}" == "8bit" ]]; then
  COMMON_ARGS+=(--load-in-8bit)
fi
if [[ "${DETERMINISTIC}" == "1" ]]; then
  COMMON_ARGS+=(--deterministic)
fi
if [[ "${ENABLE_GPU_MONITOR}" == "1" ]]; then
  COMMON_ARGS+=(--enable-gpu-monitor)
fi
if [[ "${DISABLE_AUTO_PROFILE}" == "1" ]]; then
  COMMON_ARGS+=(--disable-auto-profile)
fi
if [[ "${FORCE_PROFILE_REFRESH}" == "1" ]]; then
  COMMON_ARGS+=(--force-profile-refresh)
fi
if [[ "${REFERENCE_FORCE_REFRESH}" == "1" ]]; then
  COMMON_ARGS+=(--reference-force-refresh)
fi
if [[ "${DISABLE_ONLINE_PROFILE_UPDATE}" == "1" ]]; then
  COMMON_ARGS+=(--disable-online-profile-update)
fi
if [[ -n "${QUESTION_LIMIT}" ]]; then
  COMMON_ARGS+=(--limit "${QUESTION_LIMIT}")
fi
if [[ "${JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD}" == "1" ]]; then
  COMMON_ARGS+=(--join-canceled-proactive-before-tree-build)
fi
if [[ "${DISABLE_PROACTIVE_BUDGET}" == "1" ]]; then
  COMMON_ARGS+=(--disable-proactive-budget)
fi

append_meta() {
  local json_line="$1"
  printf '%s\n' "${json_line}" >> "${META_JSONL}"
}

selected_result_has_algorithm() {
  local objective_metric="$1"
  local bench="$2"
  local algo_id="$3"
  local selected_csv="${SELECTED_RESULT_ROOT}/${objective_metric}/${bench}/algorithm_tps_metric_treebuild_breakdown_comparison_algorithms_latest.csv"

  [[ -f "${selected_csv}" ]] || return 1
  python3 - "$selected_csv" "$algo_id" <<'PY'
import csv
import sys

csv_path, algo_id = sys.argv[1], sys.argv[2]
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("algorithm") == algo_id and row.get("status", "ok") == "ok":
            sys.exit(0)
sys.exit(1)
PY
}

should_skip_selected_existing_autodraft() {
  local objective_metric="$1"
  local bench="$2"
  local algo_id="$3"

  [[ "${SKIP_SELECTED_EXISTING_AUTODRAFT}" == "1" ]] || return 1
  selected_result_has_algorithm "${objective_metric}" "${bench}" "${algo_id}"
}

is_metric_invariant_objective_run_enabled() {
  local objective_metric="$1"

  [[ "${RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE}" == "1" ]] || return 0
  [[ "${objective_metric}" == "${PRIMARY_OBJECTIVE_METRIC}" ]]
}

should_skip_server_only_autodraft_for_objective() {
  local objective_metric="$1"

  [[ "${SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY}" == "1" ]] || return 1
  [[ "${objective_metric}" == "draft_energy" ]]
}

ablation_suffix_for_mode() {
  local ablation_mode="$1"
  case "${ablation_mode}" in
    full)
      printf ''
      ;;
    no_proactive)
      printf '_no_proactive'
      ;;
    adaptive_only)
      printf '_adaptive_only'
      ;;
    *)
      echo "[ERROR] unsupported AUTODRAFT ablation mode: ${ablation_mode}" >&2
      return 1
      ;;
  esac
}

append_ablation_args() {
  local ablation_mode="$1"
  local -n out_args="$2"
  case "${ablation_mode}" in
    full)
      out_args+=(--proactive-drafting --adaptive-proactive-threshold --proactive-threshold "${PROACTIVE_THRESHOLD}")
      ;;
    no_proactive)
      ;;
    adaptive_only)
      out_args+=(--disable-online-profile-update --disable-online-calibration)
      ;;
    *)
      echo "[ERROR] unsupported AUTODRAFT ablation mode: ${ablation_mode}" >&2
      return 1
      ;;
  esac
}

run_with_live_stage_logs() {
  local log_file="$1"
  shift
  python3 - "$log_file" "$@" <<'PY'
import re
import subprocess
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
cmd = sys.argv[2:]
log_path.parent.mkdir(parents=True, exist_ok=True)

phase = None
last_query_done = None
last_query_total = None

def set_phase(new_phase: str, reason: str) -> None:
    global phase
    if phase != new_phase:
        phase = new_phase
        print(f"[STAGE] {new_phase} :: {reason}", flush=True)

query_re = re.compile(r'(\d+)/(\d+)')
tqdm_re = re.compile(r'(\d+)%\|')

with log_path.open("w", encoding="utf-8") as logf:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    set_phase("startup", "process launched")

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        logf.write(raw_line)
        logf.flush()

        l = line.lower()
        if "Profiling data loaded" in line or "target Profiling data loaded" in line:
            set_phase("profile_load", line)
        elif "profiling start" in line or "profiling file is missing. Starting profiling" in line:
            set_phase("profile_build", line)
        elif "target profiling file is missing. Requesting automatic generation" in line:
            set_phase("profile_build_target_remote", line)
        elif "[targetprofile]" in l:
            set_phase("profile_build_target_remote", line)
        elif "[reference] loaded cached reference" in l:
            set_phase("reference_load", line)
        elif "[reference] no cache found" in l or "[reference] force refresh enabled" in l:
            set_phase("reference_build", line)
        elif "[reference] saved reference cache" in l:
            set_phase("reference_build_done", line)
        elif "[warmup]" in l:
            set_phase("warmup", line)
        elif "token generation" in line or ("%" in line and "it/s" in line and tqdm_re.search(line)):
            set_phase("main_experiment", "decoding in progress")

        # profile width .
        if (
            phase in {"profile_build", "profile_build_target_remote"}
            and ("profiling width=" in line or "[TargetProfile]" in line)
        ):
            print(f"[PROFILE] {line}", flush=True)

        # tqdm-like progress extraction: "...  3/80 [..]"
        if ("it/s" in line or "token generation" in line or tqdm_re.search(line)):
            m = query_re.search(line)
            if m:
                done = int(m.group(1))
                total = int(m.group(2))
                if total > 0 and (done != last_query_done or total != last_query_total):
                    last_query_done, last_query_total = done, total
                    print(f"[QUERY] {done}/{total}", flush=True)

        if "[warn]" in l or "error" in l or "traceback" in l:
            print(f"[LOG] {line}", flush=True)

    rc = proc.wait()
    print(f"[STAGE] finished :: rc={rc}", flush=True)
    sys.exit(rc)
PY
}

run_case() {
  local bench="$1"
  local algo_id="$2"
  local cost_sensitivity="$3"
  shift 3
  local extra_args=("$@")

  local objective_metric="${CURRENT_OBJECTIVE_METRIC:-${OBJECTIVE_METRIC}}"
  local answer_file="${ANSWER_DIR}/answer_${bench}_${algo_id}_${objective_metric}_${RUN_TS}.jsonl"
  local log_file="${LOG_DIR}/log_${bench}_${algo_id}_${objective_metric}_${RUN_TS}.txt"

  local cmd=(python3 -u -m "${DRAFT_ENTRYPOINT}")
  cmd+=("${COMMON_ARGS[@]}")
  cmd+=(--objective-metric "${objective_metric}")
  cmd+=(--bench-name "${bench}")
  cmd+=(--answer-file "${answer_file}")
  cmd+=(--cost-sensitivity "${cost_sensitivity}")
  cmd+=("${extra_args[@]}")

  local meta_line
  meta_line=$(
    cat <<EOF
{"timestamp":"${RUN_TS}","bench_name":"${bench}","algorithm_id":"${algo_id}","objective_metric":"${objective_metric}","cost_sensitivity":${cost_sensitivity},"ablation_mode":"${CURRENT_AUTODRAFT_ABLATION_MODE:-full}","base_model_path":"${BASE_MODEL_PATH}","draft_model_path":"${DRAFT_MODEL_PATH}","target_quantization":"${TARGET_QUANTIZATION}","draft_quantization":"${DRAFT_QUANTIZATION}","answer_file":"${answer_file}","log_file":"${log_file}"}
EOF
  )
  append_meta "${meta_line}"

  echo "[RUN] bench=${bench} algo=${algo_id} objective=${objective_metric} cs=${cost_sensitivity}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] %q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  set +e
  run_with_live_stage_logs "${log_file}" "${cmd[@]}"
  local rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "[WARN] failed: bench=${bench}, algo=${algo_id}, rc=${rc}, log=${log_file}"
  fi
  return ${rc}
}

echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] config_xml=${CONFIG_XML:-none}"
echo "[INFO] target=${TARGET_HOST}:${TARGET_PORT}"
echo "[INFO] base=${BASE_MODEL_PATH}"
echo "[INFO] draft=${DRAFT_MODEL_PATH}"
echo "[INFO] draft_entrypoint=${DRAFT_ENTRYPOINT}"
echo "[INFO] benches=${BENCHES[*]}"
echo "[INFO] objectives=${OBJECTIVE_METRICS[*]}"
echo "[INFO] primary_objective=${PRIMARY_OBJECTIVE_METRIC}"
echo "[INFO] algorithms=server_only_ar=${ENABLE_SERVER_ONLY_AR} server_only_non_tree=${ENABLE_SERVER_ONLY_NON_TREE_SD} server_only_autodraft_cs=${SERVER_ONLY_AUTODRAFT_CS_LIST} hybrid_non_tree=${ENABLE_HYBRID_NON_TREE_SD} hybrid_opt_tree=${ENABLE_HYBRID_OPT_TREE} hybrid_autodraft_cs=${AUTODRAFT_CS_LIST}"
echo "[INFO] autodraft_ablation_modes=${AUTODRAFT_ABLATION_MODES[*]}"
echo "[INFO] draft_electricity_cost_per_kwh=${DRAFT_ELECTRICITY_COST_PER_KWH} user_comm_cost_per_gb=${USER_COMM_COST_PER_GB} cloud_outbound_cost_per_gb=${CLOUD_OUTBOUND_COST_PER_GB}"
echo "[INFO] seed=${SEED} deterministic=${DETERMINISTIC}"
echo "[INFO] disable_auto_profile=${DISABLE_AUTO_PROFILE} disable_online_profile_update=${DISABLE_ONLINE_PROFILE_UPDATE}"
echo "[INFO] force_profile_refresh=${FORCE_PROFILE_REFRESH} reference_force_refresh=${REFERENCE_FORCE_REFRESH}"
echo "[INFO] enable_server_only_ar=${ENABLE_SERVER_ONLY_AR} enable_server_only_non_tree_sd=${ENABLE_SERVER_ONLY_NON_TREE_SD} enable_server_only_autodraft=${ENABLE_SERVER_ONLY_AUTODRAFT} enable_hybrid_non_tree_sd=${ENABLE_HYBRID_NON_TREE_SD} enable_hybrid_opt_tree=${ENABLE_HYBRID_OPT_TREE} enable_hybrid_autodraft=${ENABLE_HYBRID_AUTODRAFT} enable_server_only_opt_tree=${ENABLE_SERVER_ONLY_OPT_TREE}"
echo "[INFO] server_only_ar_turn_rpc=${SERVER_ONLY_AR_TURN_RPC}"
echo "[INFO] join_canceled_proactive_before_tree_build=${JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD}"
echo "[INFO] disable_proactive_budget=${DISABLE_PROACTIVE_BUDGET}"
echo "[INFO] proposed_nodes=${PROPOSED_NODES} proposed_max_depth=${PROPOSED_MAX_DEPTH} draft_profile_width_list=${DRAFT_PROFILE_WIDTH_LIST} target_profile_node_list=${TARGET_PROFILE_NODE_LIST}"
echo "[INFO] reference_cs_curve_rounds=${REFERENCE_CS_CURVE_ROUNDS} reference_max_steps_limit=${REFERENCE_MAX_STEPS_LIMIT}"
echo "[INFO] server_draft_profile_force_refresh=${SERVER_DRAFT_PROFILE_FORCE_REFRESH}"
echo "[INFO] run_metric_invariant_only_on_first_objective=${RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE} skip_server_only_autodraft_draft_energy=${SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY}"
echo "[INFO] skip_selected_existing_autodraft=${SKIP_SELECTED_EXISTING_AUTODRAFT}"

for objective_metric in "${OBJECTIVE_METRICS[@]}"; do
  CURRENT_OBJECTIVE_METRIC="${objective_metric}"
for bench in "${BENCHES[@]}"; do
for ablation_mode in "${AUTODRAFT_ABLATION_MODES[@]}"; do
  CURRENT_AUTODRAFT_ABLATION_MODE="${ablation_mode}"
  ablation_suffix="$(ablation_suffix_for_mode "${ablation_mode}")"
  ablation_args=()
  append_ablation_args "${ablation_mode}" ablation_args

  server_only_common_args=(
    --force-server-only
    --bill-draft-as-target-gpu
    --server-draft-profile-model-calls-per-count "${SERVER_DRAFT_PROFILE_MODEL_CALLS_PER_COUNT}"
    --server-draft-profile-width-list "${SERVER_DRAFT_PROFILE_WIDTH_LIST}"
  )
  if [[ "${SERVER_DRAFT_PROFILE_AUTO}" != "1" ]]; then
    server_only_common_args+=(--disable-auto-server-draft-profile)
  fi
  if [[ "${SERVER_DRAFT_PROFILE_FORCE_REFRESH}" == "1" ]]; then
    server_only_common_args+=(--server-draft-profile-force-refresh)
  fi

  # 1) Server-only AR: target-side strict autoregressive model.forward path.
  if [[ "${ENABLE_SERVER_ONLY_AR}" == "1" ]]; then
    if ! is_metric_invariant_objective_run_enabled "${CURRENT_OBJECTIVE_METRIC}"; then
      echo "[SKIP] metric-invariant algorithm already covered by primary objective=${PRIMARY_OBJECTIVE_METRIC}: objective=${CURRENT_OBJECTIVE_METRIC} algo=server_only_ar${ablation_suffix}"
    else
    algo_id="server_only_ar${ablation_suffix}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    server_only_ar_args=(
      --force-server-only-ar
      --server-only-ar-max-new-tokens "${SERVER_ONLY_AR_MAX_NEW_TOKENS}"
      --bill-draft-as-target-gpu
    )
    if [[ "${SERVER_ONLY_AR_TURN_RPC}" == "1" ]]; then
      server_only_ar_args+=(--server-only-ar-turn-rpc)
    else
      server_only_ar_args+=(--server-only-ar-stream-rpc)
    fi
    run_case "${bench}" "${algo_id}" "${BASE_COST_SENSITIVITY}" "${server_only_ar_args[@]}" || true
    fi
  fi

  # 2) Server-only non-tree SD
  if [[ "${ENABLE_SERVER_ONLY_NON_TREE_SD}" == "1" ]]; then
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    server_only_non_tree_args=(
      "${server_only_common_args[@]}"
      --nodes "${NON_TREE_NODES}"
      --max_depth "${NON_TREE_DEPTH}"
      --fixed-depth
      --fixed-nnodes
      --fixed-width
      --fixed-width-value "${NON_TREE_WIDTH}"
    )
    run_case "${bench}" "server_only_non_tree_sd${ablation_suffix}" "${BASE_COST_SENSITIVITY}" "${server_only_non_tree_args[@]}" || true
  fi

 
  # 3) Server-only opt-tree
  if [[ "${ENABLE_SERVER_ONLY_OPT_TREE}" == "1" ]]; then
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    server_only_opt_tree_args=(
      "${server_only_common_args[@]}"
      --nodes "${PROPOSED_NODES}"
      --max_depth "${PROPOSED_MAX_DEPTH}"
      --opt-tree
      --fixed-nnodes
      --fixed-width
      --fixed-width-value "${PROPOSED_NODES}"
    )
    run_case "${bench}" "server_only_opt_tree${ablation_suffix}" "${BASE_COST_SENSITIVITY}" "${server_only_opt_tree_args[@]}" || true
  fi

  # 4) Server-only AutoDraft
  if [[ "${ENABLE_SERVER_ONLY_AUTODRAFT}" == "1" ]]; then
  for cs in ${SERVER_ONLY_AUTODRAFT_CS_LIST}; do
    algo_id="server_only_autodraft_cs${cs}${ablation_suffix}"
    if should_skip_server_only_autodraft_for_objective "${CURRENT_OBJECTIVE_METRIC}"; then
      echo "[SKIP] server-only AutoDraft has no user-device draft energy metric: objective=${CURRENT_OBJECTIVE_METRIC} algo=${algo_id}"
      continue
    fi
    if should_skip_selected_existing_autodraft "${CURRENT_OBJECTIVE_METRIC}" "${bench}" "${algo_id}"; then
      echo "[SKIP] selected result exists: objective=${CURRENT_OBJECTIVE_METRIC} bench=${bench} algo=${algo_id}"
      continue
    fi
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    extra_server_only_args=(
      "${server_only_common_args[@]}"
      --nodes "${PROPOSED_NODES}"
      --max_depth "${PROPOSED_MAX_DEPTH}"
    )
    extra_server_only_args+=("${ablation_args[@]}")
    run_case "${bench}" "${algo_id}" "${cs}" "${extra_server_only_args[@]}" || true
  done
  fi

  # 5) Hybrid non-tree SD
  if [[ "${ENABLE_HYBRID_NON_TREE_SD}" == "1" ]]; then
    if ! is_metric_invariant_objective_run_enabled "${CURRENT_OBJECTIVE_METRIC}"; then
      echo "[SKIP] metric-invariant algorithm already covered by primary objective=${PRIMARY_OBJECTIVE_METRIC}: objective=${CURRENT_OBJECTIVE_METRIC} algo=hybrid_non_tree_sd${ablation_suffix}"
    else
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    run_case "${bench}" "hybrid_non_tree_sd${ablation_suffix}" "${BASE_COST_SENSITIVITY}" \
      --disable-server-only \
      --nodes "${NON_TREE_NODES}" \
      --max_depth "${NON_TREE_DEPTH}" \
      --fixed-depth \
      --fixed-nnodes \
      --fixed-width \
      --fixed-width-value "${NON_TREE_WIDTH}" || true
    fi
  fi

  # 6) Hybrid opt-tree
  if [[ "${ENABLE_HYBRID_OPT_TREE}" == "1" ]]; then
    if ! is_metric_invariant_objective_run_enabled "${CURRENT_OBJECTIVE_METRIC}"; then
      echo "[SKIP] metric-invariant algorithm already covered by primary objective=${PRIMARY_OBJECTIVE_METRIC}: objective=${CURRENT_OBJECTIVE_METRIC} algo=hybrid_opt_tree${ablation_suffix}"
    else
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    run_case "${bench}" "hybrid_opt_tree${ablation_suffix}" "${BASE_COST_SENSITIVITY}" \
      --disable-server-only \
      --nodes "${PROPOSED_NODES}" \
      --max_depth "${PROPOSED_MAX_DEPTH}" \
      --opt-tree \
      --fixed-nnodes \
      --fixed-width \
      --fixed-width-value "${PROPOSED_NODES}" || true
    fi
  fi

  # 7~9) Hybrid AutoDraft cs=0/0.5/1
  if [[ "${ENABLE_HYBRID_AUTODRAFT}" == "1" ]]; then
  for cs in ${AUTODRAFT_CS_LIST}; do
    algo_id="autodraft_cs${cs}${ablation_suffix}"
    if should_skip_selected_existing_autodraft "${CURRENT_OBJECTIVE_METRIC}" "${bench}" "${algo_id}"; then
      echo "[SKIP] selected result exists: objective=${CURRENT_OBJECTIVE_METRIC} bench=${bench} algo=${algo_id}"
      continue
    fi
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    extra_hybrid_args=(
      --disable-server-only
      --nodes "${PROPOSED_NODES}"
      --max_depth "${PROPOSED_MAX_DEPTH}"
    )
    extra_hybrid_args+=("${ablation_args[@]}")
    run_case "${bench}" "${algo_id}" "${cs}" \
      "${extra_hybrid_args[@]}" || true
  done
  fi

done

  if [[ "${SKIP_SLEEP}" != "1" ]]; then
    echo "[INFO] bench ${bench} done. sleep ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
  fi

done
done

bench_json_array="$(printf '"%s",' "${BENCHES[@]}")"
bench_json_array="[${bench_json_array%,}]"
ablation_mode_json_array="$(printf '"%s",' "${AUTODRAFT_ABLATION_MODES[@]}")"
ablation_mode_json_array="[${ablation_mode_json_array%,}]"
algorithm_json=""
for ablation_mode in "${AUTODRAFT_ABLATION_MODES[@]}"; do
  ablation_suffix="$(ablation_suffix_for_mode "${ablation_mode}")"
  if [[ "${ENABLE_SERVER_ONLY_AR}" == "1" ]]; then
    algorithm_json="${algorithm_json}\"server_only_ar${ablation_suffix}\","
  fi
  if [[ "${ENABLE_SERVER_ONLY_NON_TREE_SD}" == "1" ]]; then
    algorithm_json="${algorithm_json}\"server_only_non_tree_sd${ablation_suffix}\","
  fi
  if [[ "${ENABLE_SERVER_ONLY_OPT_TREE}" == "1" ]]; then
    algorithm_json="${algorithm_json}\"server_only_opt_tree${ablation_suffix}\","
  fi
  if [[ "${ENABLE_HYBRID_NON_TREE_SD}" == "1" ]]; then
    algorithm_json="${algorithm_json}\"hybrid_non_tree_sd${ablation_suffix}\","
  fi
  if [[ "${ENABLE_HYBRID_OPT_TREE}" == "1" ]]; then
    algorithm_json="${algorithm_json}\"hybrid_opt_tree${ablation_suffix}\","
  fi
  for cs in ${AUTODRAFT_CS_LIST}; do
    if [[ "${ENABLE_HYBRID_AUTODRAFT}" == "1" ]]; then
      algorithm_json="${algorithm_json}\"autodraft_cs${cs}${ablation_suffix}\","
    fi
  done
  if [[ "${ENABLE_SERVER_ONLY_AUTODRAFT}" == "1" ]]; then
    for cs in ${SERVER_ONLY_AUTODRAFT_CS_LIST}; do
      algorithm_json="${algorithm_json}\"server_only_autodraft_cs${cs}${ablation_suffix}\","
    done
  fi
done
algorithm_json="[${algorithm_json%,}]"

cat > "${SUMMARY_JSON}" <<EOF
{
  "timestamp": "${RUN_TS}",
  "config_xml": "${CONFIG_XML}",
  "target_host": "${TARGET_HOST}",
  "target_port": ${TARGET_PORT},
  "base_model_path": "${BASE_MODEL_PATH}",
  "draft_model_path": "${DRAFT_MODEL_PATH}",
  "target_quantization": "${TARGET_QUANTIZATION}",
  "draft_quantization": "${DRAFT_QUANTIZATION}",
  "draft_electricity_cost_per_kwh": ${DRAFT_ELECTRICITY_COST_PER_KWH},
  "user_communication_cost_per_gb": ${USER_COMM_COST_PER_GB},
  "cloud_outbound_cost_per_gb": ${CLOUD_OUTBOUND_COST_PER_GB},
  "objective_metric": "${OBJECTIVE_METRIC}",
  "objective_metrics": [$(printf '"%s",' "${OBJECTIVE_METRICS[@]}" | sed 's/,$//')],
  "primary_objective_metric": "${PRIMARY_OBJECTIVE_METRIC}",
  "benches": ${bench_json_array},
  "algorithms": ${algorithm_json},
  "autodraft_ablation_modes": ${ablation_mode_json_array},
  "proposed_nodes": ${PROPOSED_NODES},
  "proposed_max_depth": ${PROPOSED_MAX_DEPTH},
  "draft_profile_width_list": "${DRAFT_PROFILE_WIDTH_LIST}",
  "target_profile_node_list": "${TARGET_PROFILE_NODE_LIST}",
  "run_metric_invariant_only_on_first_objective": ${RUN_METRIC_INVARIANT_ONLY_ON_FIRST_OBJECTIVE},
  "skip_server_only_autodraft_draft_energy": ${SKIP_SERVER_ONLY_AUTODRAFT_DRAFT_ENERGY},
  "force_profile_refresh": ${FORCE_PROFILE_REFRESH},
  "reference_force_refresh": ${REFERENCE_FORCE_REFRESH},
  "join_canceled_proactive_before_tree_build": ${JOIN_CANCELED_PROACTIVE_BEFORE_TREE_BUILD},
  "disable_proactive_budget": ${DISABLE_PROACTIVE_BUDGET},
  "total_runs": ${TOTAL_RUNS},
  "success_count": ${SUCCESS_COUNT},
  "fail_count": ${FAIL_COUNT},
  "meta_jsonl": "${META_JSONL}",
  "answer_dir": "${ANSWER_DIR}",
  "log_dir": "${LOG_DIR}"
}
EOF

echo "[DONE] summary=${SUMMARY_JSON}"
echo "[DONE] meta=${META_JSONL}"

