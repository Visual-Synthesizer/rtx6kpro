#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-ds4-v10-vllm61f32d0-b12x90172a5-fi7176f85-cu132-20260710}"
export OUT="${OUT:-/root/bench-results/ds4-v10-$(date -u +%Y%m%d-%H%M%S)}"
export PROGRESS_FILE="${PROGRESS_FILE:-${OUT}/progress.log}"
export LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/run-ds4-v10-server.sh}"
export SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/run-ds4-v10-sweep.sh}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-/root/rtx6kpro/.no-vllm-runtime-patch-for-ds4-v10}"
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-ds4-v10}"

# This v10 sweep is intentionally restricted to GPUs 0-7. The GLM service on
# GPUs 8-15 is outside its allocation and cannot be selected by the scheduler.
export GPU_GROUPS_TP2="0,1 2,3 4,5 6,7"
export GPU_GROUPS_TP4="0,1,2,3 4,5,6,7"
export SYNC_WAVE_READY=1

exec "${SCRIPT_DIR}/run-ds4-v9-sweep.sh" "$@"
