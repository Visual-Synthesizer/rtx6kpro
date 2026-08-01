#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE='voipmonitor/vllm:gilded-gnosis-v20-vllm6634ec9-sib2bff71-fi801d57a-cu132-20260801-r19'
export EXPECTED_IMAGE_ID='sha256:01de59d45432fb3fe169abf6ef10293712a4dc262d2a16c0e3d56b0b6bcf2ab7'
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
