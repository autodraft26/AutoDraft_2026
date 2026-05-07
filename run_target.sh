#!/usr/bin/env bash
set -euo pipefail

# Load .env from repository root when present.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Usage:
#   ./run_target.sh
#   TARGET_HOST=127.0.0.1 TARGET_PORT=26001 BASE_MODEL_PATH=meta-llama/Llama-3.3-70B-Instruct ./run_target.sh
#   ./run_target.sh --host 127.0.0.1 --port 26001 --base-model-path meta-llama/Llama-3.3-70B-Instruct --draft-model-path meta-llama/Llama-3.2-3B-Instruct

HOST="${TARGET_HOST:-${AUTODRAFT_TARGET_HOST:-${AUTODRAFT_HOST:-192.168.0.12}}}"
PORT="${TARGET_PORT:-26001}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
SERVER_NAME="${SERVER_NAME:-${AUTODRAFT_SERVER_NAME:-target}}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RUNS="${RUNS:-1}"
LOAD_FLAG="${LOAD_FLAG:---load-in-8bit}"
EAGER_LOAD="${EAGER_LOAD:-0}"
AUTO_TARGET_PROFILE="${AUTO_TARGET_PROFILE:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
    --draft-model-path) DRAFT_MODEL_PATH="$2"; shift 2 ;;
    --server-name) SERVER_NAME="$2"; shift 2 ;;
    --device-map) DEVICE_MAP="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --load-in-8bit) LOAD_FLAG="--load-in-8bit"; shift ;;
    --load-in-4bit) LOAD_FLAG="--load-in-4bit"; shift ;;
    --eager-load) EAGER_LOAD=1; shift ;;
    --lazy-load) EAGER_LOAD=0; shift ;;
    --disable-auto-target-profile) AUTO_TARGET_PROFILE=0; shift ;;
    --enable-auto-target-profile) AUTO_TARGET_PROFILE=1; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

for i in $(seq 1 "$RUNS"); do
  echo "===== target run $i / $RUNS ====="
  EXTRA_FLAGS=()
  if [[ "$EAGER_LOAD" == "1" ]]; then
    EXTRA_FLAGS+=("--eager-load")
  fi
  if [[ "$AUTO_TARGET_PROFILE" == "1" ]]; then
    EXTRA_FLAGS+=("--enable-auto-target-profile")
  else
    EXTRA_FLAGS+=("--disable-auto-target-profile")
  fi
  python3 -m evaluation.eval_autodraft_target \
    --host "$HOST" \
    --port "$PORT" \
    --base-model-path "$BASE_MODEL_PATH" \
    --draft-model-path "$DRAFT_MODEL_PATH" \
    --server-name "$SERVER_NAME" \
    "$LOAD_FLAG" \
    --device-map "$DEVICE_MAP" \
    --temperature 0.0 \
    --enable-gpu-monitor \
    --output-file /dev/null \
    --seed 4 \
    "${EXTRA_FLAGS[@]}"
  echo "===== run $i / $RUNS finished (exit: $?) ====="
done

echo "All $RUNS runs completed."

# =========================
# Supported target/draft pairs
# =========================
# Llama:
#   target meta-llama/Llama-3.3-70B-Instruct
#     draft meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct
# Qwen2.5:
#   target Qwen/Qwen2.5-14B-Instruct
#     draft Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct
#   target Qwen/Qwen2.5-32B-Instruct
#     draft Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct
# Qwen3:
#   target Qwen/Qwen3-14B
#     draft Qwen/Qwen3-0.6B
#   target Qwen/Qwen3-32B
#     draft Qwen/Qwen3-0.6B
