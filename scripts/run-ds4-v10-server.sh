#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709}"
export NAME="${NAME:-ds4-v10}"
export CACHE="${CACHE:-/root/.cache/vllm-ds4-v10/${NAME}}"

exec "${SCRIPT_DIR}/run-ds4-v9-server.sh" "$@"
