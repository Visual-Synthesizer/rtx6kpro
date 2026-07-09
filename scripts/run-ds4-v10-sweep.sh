#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709}"
export OUT="${OUT:-/root/bench-results/ds4-v10-ff-$(date -u +%Y%m%d-%H%M%S)}"
export PROGRESS_FILE="${PROGRESS_FILE:-${OUT}/progress.log}"
export LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/run-ds4-v10-server.sh}"
export SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/run-ds4-v10-sweep.sh}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-/root/rtx6kpro/.no-vllm-runtime-patch-for-ds4-v10}"

exec "${SCRIPT_DIR}/run-ds4-v9-sweep.sh" "$@"
