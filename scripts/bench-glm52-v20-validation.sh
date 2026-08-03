#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE='voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-sibbbdccc-fi801d57a-cu132-20260803-r26'
export EXPECTED_IMAGE_ID='sha256:30a752f0b490a3841610dfbea8f5758eab66924537feaeb1cd06b8e6ac92cce4'
export RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v20-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
export CACHE_A="${CACHE_A:-/root/.cache/vllm-glm52-release/slot-a}"
export CACHE_B="${CACHE_B:-/root/.cache/vllm-glm52-release/slot-b}"
export TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp/glm52-v20-validation}"
export RELEASE_LABEL="${RELEASE_LABEL:-v20-r26}"
export NAME_PREFIX="${NAME_PREFIX:-glm52-v20-r26-validation}"
if [[ -z "${ONLINE_MXFP8_CONFIG_JSON+x}" ]]; then
  export ONLINE_MXFP8_CONFIG_JSON='{"linear":{"weight":"mxfp8"}}'
fi

exec "${script_dir}/bench-glm52-v18-validation.sh" "$@"
