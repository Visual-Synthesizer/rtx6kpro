#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/compose/glm52-v15.yml}"
export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v19-vllm0d1ad03-b12x90172a5-cu132-20260709}"
export NAME="${NAME:-glm52-v15}"
export CACHE_ROOT="${CACHE_ROOT:-/root/.cache/vllm-glm52-v15}"
export CONTAINER_LAUNCHER="${CONTAINER_LAUNCHER:-/usr/local/bin/run-glm52-v15-server}"
export GPU_DEVICE_REQUESTS="${GPU_DEVICE_REQUESTS:-1}"

exec "${ROOT_DIR}/scripts/run-glm52-v14-compose.sh" "$@"
