#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Jetson Orin uses the jtop-backed in-process monitor wrapper. The wrapper
# patches only GPUMonitor before delegating to the regular draft entrypoint.
export DRAFT_ENTRYPOINT="${DRAFT_ENTRYPOINT:-evaluation.eval_autodraft_draft_orin}"
export DRAFT_DEVICE_NAME="${DRAFT_DEVICE_NAME:-orin}"

# Some Orin virtualenvs install CUDA shared libraries under site-packages but
# do not add that directory to the dynamic linker path.
CUDA_PY_LIB_DIR="${ROOT_DIR}/.venv/lib/python3.10/site-packages/nvidia/cu12/lib"
if [[ -d "${CUDA_PY_LIB_DIR}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${CUDA_PY_LIB_DIR}:"*) ;;
    *) export LD_LIBRARY_PATH="${CUDA_PY_LIB_DIR}:${LD_LIBRARY_PATH:-}" ;;
  esac
fi

exec bash "${SCRIPT_DIR}/run_main_experiment_overall_performance.sh" "$@"
