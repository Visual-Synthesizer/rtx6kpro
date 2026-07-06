#!/usr/bin/env bash
set -euo pipefail

# Reproducible GLM-5.2 v14 CUDA 13.2 image build.
#
# Stage 1 builds the normal vLLM+B12X image from the blackwell-llm-docker
# Dockerfile using pinned source commits. The vLLM ref already contains the
# FP8 bridge and MXFP4 online dense overlay changes, so no local vLLM patch is
# applied here.
# Stage 2 only adds the GLM 5.2 launcher and makes buffered InstantTensor
# loading the image default.

BLACKWELL_DOCKER_DIR="${BLACKWELL_DOCKER_DIR:-/root/vllm/blackwell-llm-docker}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GLM52_V14_SERVER="${RUN_GLM52_V14_SERVER:-${SCRIPT_DIR}/run-glm52-v14-server}"

DATE_TAG="${DATE_TAG:-20260706}"
VLLM_BRANCH_TAG="modelopt-online-fp8-overlay"
IMAGE_FLAVOR_TAG="${IMAGE_FLAVOR_TAG:-glm52-v14-online-fp8-mxfp8-v2}"
VLLM_REF="${VLLM_REF:-codex/modelopt-online-fp8-overlay-20260706}"
VLLM_COMMIT="${VLLM_COMMIT:-02f5b41befc6da0db8ad4d79aa02bab1156de7b2}"
B12X_COMMIT="${B12X_COMMIT:-e44cb77777a075790ebe9f7aa9f225d073aea109}"
INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-${INSTANTTENSOR_COMMIT}}"

BASE_IMAGE="${BASE_IMAGE:-voipmonitor/vllm:${IMAGE_FLAVOR_TAG}-vllm${VLLM_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-it${INSTANTTENSOR_COMMIT:0:7}-nccl2304-cu132-${DATE_TAG}-base}"
FINAL_IMAGE="${FINAL_IMAGE:-voipmonitor/vllm:${IMAGE_FLAVOR_TAG}-vllm${VLLM_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-it${INSTANTTENSOR_COMMIT:0:7}-nccl2304-cu132-${DATE_TAG}}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"
SKIP_BASE_BUILD="${SKIP_BASE_BUILD:-0}"

if [[ "${SKIP_BASE_BUILD}" == "1" ]]; then
  echo "Skipping base image build; verifying existing base image: ${BASE_IMAGE}"
  docker image inspect "${BASE_IMAGE}" >/dev/null
else
  echo "Building base image: ${BASE_IMAGE}"
  (
    cd "${BLACKWELL_DOCKER_DIR}"
    export IMAGE="${BASE_IMAGE}"
    export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
    export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260626}"
    export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
    export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
    export MAX_JOBS="${MAX_JOBS:-64}"
    export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
    export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
    export PIN_SOURCE_COMMITS=1

    export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-5a73a36a7169ec5533ba474bb9204bed765dd297}"
    export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
    export FLASHINFER_REF="${FLASHINFER_REF:-${FLASHINFER_COMMIT}}"
    export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

    export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
    export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
    export DEEPGEMM_REF="${DEEPGEMM_REF:-${DEEPGEMM_COMMIT}}"

    export B12X_REPO="${B12X_REPO:-https://github.com/local-inference-lab/b12x.git}"
    export B12X_REF="${B12X_REF:-master}"
    export B12X_COMMIT="${B12X_COMMIT}"

    export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
    export VLLM_REF="${VLLM_REF}"
    export VLLM_COMMIT="${VLLM_COMMIT}"
    export VLLM_PATCH_URL="${VLLM_PATCH_URL:-}"
    export VLLM_PATCH_SHA256="${VLLM_PATCH_SHA256:-}"
    export VLLM_PATCH_FILE="${VLLM_PATCH_FILE:-}"
    export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+${VLLM_BRANCH_TAG//-/.}.${VLLM_COMMIT:0:7}.b12x${B12X_COMMIT:0:7}.cu132.${DATE_TAG}}"

    export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
    export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
    export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"

    export CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
    export CUTLASS_REF="${CUTLASS_REF:-d80a4e53b52b42550659a8696dab32705265e324}"
    export CUTLASS_COMMIT="${CUTLASS_COMMIT:-d80a4e53b52b42550659a8696dab32705265e324}"
    export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.6}"

    # The overlay below is the canonical InstantTensor install. These exports are
    # harmless for older blackwell Dockerfiles and pin the same source if the
    # local Dockerfile already supports InstantTensor build args.
    export INSTANTTENSOR_REF="${INSTANTTENSOR_REF}"
    export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT}"

    ./build-vllm-b12x-cu132.sh "$@"
  )
fi

echo "Building final GLM 5.2 launcher overlay: ${FINAL_IMAGE}"
overlay_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${overlay_dir}"
}
trap cleanup EXIT

install -m 0755 "${RUN_GLM52_V14_SERVER}" "${overlay_dir}/run-glm52-v14-server"

cat > "${overlay_dir}/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260626
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

COPY run-glm52-v14-server /usr/local/bin/run-glm52-v14-server
RUN bash -n /usr/local/bin/run-glm52-v14-server \
 && /opt/venv/bin/python - <<'PY'
import importlib.metadata as md
import instanttensor
import torch

print("instanttensor", md.version("instanttensor"), instanttensor.__file__)
print("torch.cuda.nccl.version", torch.cuda.nccl.version())
PY

ENV INSTANTTENSOR_BACKEND=BUFFERED

LABEL local-inference.instanttensor.backend_default="BUFFERED" \
      local-inference.glm52.launcher="/usr/local/bin/run-glm52-v14-server"
DOCKERFILE

DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --progress=plain \
  -f "${overlay_dir}/Dockerfile" \
  -t "${FINAL_IMAGE}" \
  "${overlay_dir}"

docker image inspect "${FINAL_IMAGE}" >/dev/null
docker run --rm -i --entrypoint /opt/venv/bin/python "${FINAL_IMAGE}" - <<'PY'
import importlib.metadata as md
import instanttensor
import vllm

print("vllm", getattr(vllm, "__version__", "unknown"))
print("instanttensor", md.version("instanttensor"), instanttensor.__file__)
PY

if [[ "${PUSH_IMAGE}" == "1" ]]; then
  docker push "${FINAL_IMAGE}"
fi

echo "Final image: ${FINAL_IMAGE}"
