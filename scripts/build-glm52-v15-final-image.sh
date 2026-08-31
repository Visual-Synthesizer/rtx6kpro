#!/usr/bin/env bash
set -euo pipefail

# Reproducible GLM-5.2 v15 CUDA 13.2 image build over
# local-inference-lab/vllm dev/fathomless-firmament plus latest b12x.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATE_TAG="${DATE_TAG:-20260709}"
export VLLM_BRANCH_TAG="${VLLM_BRANCH_TAG:-fathomless-firmament}"
export IMAGE_FLAVOR_TAG="${IMAGE_FLAVOR_TAG:-fathomless-firmament-v19}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v16-all-prs-20260709}"
export VLLM_COMMIT="${VLLM_COMMIT:-0d1ad037bd1be895f552c596f2fb8507c2584494}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/ff-v15-cute-compile-fallback-20260709}"
export B12X_COMMIT="${B12X_COMMIT:-90172a504e96d246e07cb1ebad3b291532445560}"

export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-${INSTANTTENSOR_COMMIT}}"

export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+fathomless.firmament.${VLLM_COMMIT:0:7}.b12x${B12X_COMMIT:0:7}.cu132.${DATE_TAG}}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.1.0}"

export BASE_IMAGE="${BASE_IMAGE:-voipmonitor/vllm:${IMAGE_FLAVOR_TAG}-vllm${VLLM_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-cu132-${DATE_TAG}-base}"
export FINAL_IMAGE="${FINAL_IMAGE:-voipmonitor/vllm:${IMAGE_FLAVOR_TAG}-vllm${VLLM_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-cu132-${DATE_TAG}}"

export RUN_GLM52_SERVER="${RUN_GLM52_SERVER:-${SCRIPT_DIR}/run-glm52-v14-server}"
export OVERLAY_LAUNCHER_NAME="${OVERLAY_LAUNCHER_NAME:-run-glm52-v15-server}"

exec "${SCRIPT_DIR}/build-glm52-v14-final-image.sh" "$@"
