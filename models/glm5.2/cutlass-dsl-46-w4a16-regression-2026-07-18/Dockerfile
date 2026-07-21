ARG BASE_IMAGE=voipmonitor/vllm:gilded-gnosis-v19-vllm975c9e4-b12xbc85ef3-fi801d57a-cu132-20260718
FROM ${BASE_IMAGE}

ARG CUTLASS_DSL_VERSION=4.5.3

RUN /opt/venv/bin/pip uninstall -y \
      nvidia-cutlass-dsl \
      nvidia-cutlass-dsl-libs-base \
      nvidia-cutlass-dsl-libs-core \
      nvidia-cutlass-dsl-libs-cu12 \
      nvidia-cutlass-dsl-libs-cu13 \
    && /opt/venv/bin/pip install --no-deps \
      "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
      "nvidia-cutlass-dsl-libs-base==${CUTLASS_DSL_VERSION}" \
    && /opt/venv/bin/pip install --force-reinstall --no-deps \
      "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}" \
    && CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
      /opt/venv/bin/python - <<'PY'
import importlib.metadata as md
import os

import cutlass.cute as cute

version = os.environ["CUTLASS_DSL_VERSION"]
assert md.version("nvidia-cutlass-dsl") == version
assert md.version("nvidia-cutlass-dsl-libs-base") == version
assert md.version("nvidia-cutlass-dsl-libs-cu13") == version
assert hasattr(cute.nvgpu.warp, "MmaMXF8Op")
PY
