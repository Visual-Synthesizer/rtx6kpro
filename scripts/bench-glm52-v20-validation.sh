#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm5517197-sibe0edca-fi801d57a-cu132-20260725}"
export EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:727ac3af71b729be93fd01c9fe60472c27f619c8ef0e3c67b4f627023933027c}"
export RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v20-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
export CACHE_A="${CACHE_A:-/root/.cache/vllm-glm52-release/slot-a}"
export CACHE_B="${CACHE_B:-/root/.cache/vllm-glm52-release/slot-b}"
export TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp/glm52-v20-validation}"
export RELEASE_LABEL="${RELEASE_LABEL:-v20}"
export NAME_PREFIX="${NAME_PREFIX:-glm52-v20-validation}"
if [[ -z "${ONLINE_MXFP8_CONFIG_JSON+x}" ]]; then
  export ONLINE_MXFP8_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}'
fi

exec "${script_dir}/bench-glm52-v18-validation.sh" "$@"
