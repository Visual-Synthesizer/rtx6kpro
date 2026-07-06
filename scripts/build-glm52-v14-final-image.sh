#!/usr/bin/env bash
set -euo pipefail

# Reproducible GLM-5.2 v14 CUDA 13.2 image build.
#
# Stage 1 builds the normal vLLM+B12X image from the blackwell-llm-docker
# Dockerfile using pinned source commits and the fp8.py bridge patch.
# Stage 2 overlays InstantTensor from pinned latest git and makes buffered
# InstantTensor loading the image default.

BLACKWELL_DOCKER_DIR="${BLACKWELL_DOCKER_DIR:-/root/vllm/blackwell-llm-docker}"

DATE_TAG="${DATE_TAG:-20260706}"
VLLM_BRANCH_TAG="dev-eldritch-enlightenment"
VLLM_REF="${VLLM_REF:-dev/eldritch-enlightenment}"
VLLM_COMMIT="${VLLM_COMMIT:-c382f1d28d5be2f867c216609408bdb424d6049a}"
FP8_PATCH_COMMIT="${FP8_PATCH_COMMIT:-d00593416aeb3925553ccd589d91df7075d618f6}"
FP8_PATCH_SHA256="${FP8_PATCH_SHA256:-68fc9230dfcbf08a7e599fb201a18784fa67900731c55d99d780a1b07547049a}"
B12X_COMMIT="${B12X_COMMIT:-e44cb77777a075790ebe9f7aa9f225d073aea109}"
INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-${INSTANTTENSOR_COMMIT}}"

BASE_IMAGE="${BASE_IMAGE:-voipmonitor/vllm:${VLLM_BRANCH_TAG}-vllm${VLLM_COMMIT:0:7}-fp8${FP8_PATCH_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-cu132-${DATE_TAG}-base}"
FINAL_IMAGE="${FINAL_IMAGE:-voipmonitor/vllm:${VLLM_BRANCH_TAG}-vllm${VLLM_COMMIT:0:7}-fp8${FP8_PATCH_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}-it${INSTANTTENSOR_COMMIT:0:7}-cu132-${DATE_TAG}}"
PUSH_IMAGE="${PUSH_IMAGE:-0}"

FP8_PATCH_URL="${FP8_PATCH_URL:-https://github.com/local-inference-lab/vllm/commit/${FP8_PATCH_COMMIT}.patch}"

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
  export VLLM_PATCH_URL="${FP8_PATCH_URL}"
  export VLLM_PATCH_SHA256="${FP8_PATCH_SHA256}"
  export VLLM_PATCH_FILE=""
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev279+dev.eldritch.enlightenment.${VLLM_COMMIT:0:7}.fp8${FP8_PATCH_COMMIT:0:7}.b12x${B12X_COMMIT:0:7}.cu132.${DATE_TAG}}"

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

echo "Building final InstantTensor overlay: ${FINAL_IMAGE}"
overlay_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${overlay_dir}"
}
trap cleanup EXIT

cat > "${overlay_dir}/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260626
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ARG INSTANTTENSOR_REPO=https://github.com/scitix/InstantTensor.git
ARG INSTANTTENSOR_REF=main
ARG INSTANTTENSOR_COMMIT

RUN git clone --recursive "${INSTANTTENSOR_REPO}" /tmp/instanttensor-src \
 && cd /tmp/instanttensor-src \
 && if [[ "${INSTANTTENSOR_REF}" =~ ^[0-9a-f]{40}$ ]]; then \
      git fetch --depth=1 origin "${INSTANTTENSOR_REF}"; \
      git checkout FETCH_HEAD; \
    else \
      git checkout "${INSTANTTENSOR_REF}"; \
    fi \
 && git submodule update --init --recursive \
 && if [[ -n "${INSTANTTENSOR_COMMIT}" ]]; then [[ "$(git rev-parse HEAD)" = "${INSTANTTENSOR_COMMIT}" ]] || { echo "ERROR: INSTANTTENSOR_COMMIT mismatch: HEAD=$(git rev-parse HEAD) expected=${INSTANTTENSOR_COMMIT}" >&2; exit 1; }; fi \
 && /opt/venv/bin/python -m pip install --no-build-isolation --no-deps --force-reinstall . \
 && cd / \
 && rm -rf /tmp/instanttensor-src \
 && /opt/venv/bin/python - <<'PY'
import importlib.metadata as md
import instanttensor

print("instanttensor", md.version("instanttensor"), instanttensor.__file__)
PY

ENV INSTANTTENSOR_BACKEND=BUFFERED

LABEL local-inference.instanttensor.repo="${INSTANTTENSOR_REPO}" \
      local-inference.instanttensor.branch="${INSTANTTENSOR_REF}" \
      local-inference.instanttensor.commit="${INSTANTTENSOR_COMMIT}" \
      local-inference.instanttensor.backend_default="BUFFERED"
DOCKERFILE

DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}" \
  --build-arg INSTANTTENSOR_REF="${INSTANTTENSOR_REF}" \
  --build-arg INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT}" \
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
