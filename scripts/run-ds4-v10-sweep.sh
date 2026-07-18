#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-ds4-v10-vllmadf15ca-b12x90172a5-fi2cba2f7-cu132-20260712}"
export OUT="${OUT:-/root/bench-results/ds4-v10-$(date -u +%Y%m%d-%H%M%S)}"
export PROGRESS_FILE="${PROGRESS_FILE:-${OUT}/progress.log}"
export LAUNCHER="${LAUNCHER:-${SCRIPT_DIR}/run-ds4-v10-server.sh}"
export SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/run-ds4-v10-sweep.sh}"
export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-/root/rtx6kpro/.no-vllm-runtime-patch-for-ds4-v10}"
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-ds4-v10}"

# Use the full 16-GPU host by default. Callers can still provide a smaller
# allocation without changing the synchronized load/benchmark scheduler.
export GPU_GROUPS_TP2="${GPU_GROUPS_TP2:-0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15}"
export GPU_GROUPS_TP4="${GPU_GROUPS_TP4:-0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15}"
export TPS="${TPS:-2}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_BATCHED="${MAX_BATCHED:-4096}"
export GPU_MEM="${GPU_MEM:-0.95}"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${VLLM_PREFIX_CACHE_RETENTION_INTERVAL:-4096}"
export SYNC_WAVE_READY=1

exec "${SCRIPT_DIR}/run-ds4-v9-sweep.sh" "$@"
