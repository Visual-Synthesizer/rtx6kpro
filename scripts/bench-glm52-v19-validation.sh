#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-vllmf879d86-b12xc7dc733-fi801d57a-cu132-20260719}"
export EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:5014f02f99143a16121018dcfc3cf11fce101c5edfd7e06f971f94490c839a89}"
export RESULT_ROOT="${RESULT_ROOT:-/root/bench-results/glm52-v19-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
export CACHE_A="${CACHE_A:-/root/.cache/vllm-glm52-v19/final-validation-newmaster}"
export CACHE_B="${CACHE_B:-/root/.cache/vllm-glm52-v19/final-newmaster-cache-e2e-20260718}"
export TMP_ROOT="${TMP_ROOT:-/root/vllm/tmp/glm52-v19-validation}"
export RELEASE_LABEL="${RELEASE_LABEL:-v19}"
export NAME_PREFIX="${NAME_PREFIX:-glm52-v19-validation}"

# Preserve the v18 release methodology so v18/v19 deltas remain directly
# comparable. The base runner waits for both paired servers before measuring.
exec "${script_dir}/bench-glm52-v18-validation.sh" "$@"
