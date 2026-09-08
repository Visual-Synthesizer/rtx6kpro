# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43
ARG FLASHKDA_ARTIFACT_IMAGE=voipmonitor/vllm@sha256:284784e685aa0377f1cf63a312a364fc884b02beb98949d6886624edbddb3806

FROM ${FLASHKDA_ARTIFACT_IMAGE} AS flashkda-qualified

FROM scratch AS flashkda-artifact
COPY --from=flashkda-qualified \
  /opt/glm53-flash/vllm/vllm/_flashkda_C.abi3.so \
  /_flashkda_C.abi3.so

FROM ${BASE_IMAGE} AS lmcache-build

ARG LMCACHE_VERSION=0.5.5.dev0+glm53cache.engine.822330db

COPY --from=lmcache_source / /tmp/lmcache-source
WORKDIR /tmp/lmcache-source

# Build native LMCache components against the serving image's exact Python,
# Torch, CUDA, and C++ ABI. The build stage is not part of the runtime rootfs.
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
      "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel packaging ninja \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${LMCACHE_VERSION}" \
       BUILD_WITH_CUDA=1 \
       LMCACHE_CUDA_MAJOR=13 \
       TORCH_CUDA_ARCH_LIST=12.0 \
       ENABLE_CXX11_ABI=1 \
       MAX_JOBS=2 \
       NVCC_THREADS=auto \
       /opt/venv/bin/python -m pip wheel \
         --no-build-isolation --no-deps --wheel-dir /tmp/lmcache-artifacts . \
    && make -C csrc/cumem_ipc_interposer CUDA_HOME=/usr/local/cuda \
    && install -Dm755 \
      csrc/cumem_ipc_interposer/liblmcache_cumem_shareable.so \
      /tmp/lmcache-artifacts/liblmcache_cumem_shareable.so \
    && cd /tmp/lmcache-artifacts \
    && sha256sum *.whl liblmcache_cumem_shareable.so > SHA256SUMS

FROM ${BASE_IMAGE}

ARG VLLM_SOURCE_REVISION=63a82f8d323e8538cbe6f88ae1812a1c01577a0f
ARG VLLM_SOURCE_TREE=f54cd9ca2b9434727715197d32150b75e82a9ebf
ARG VLLM_PACKAGE_TREE=a336313dfd12c6b692051d1d3291712dda0c2a3d
ARG B12X_SOURCE_REVISION=e8ad299b174f16e2e8fb5879bea272f4efbb53f2
ARG B12X_SOURCE_TREE=f3cd8a9eb00d3226a1acbbed1efedf10cc1c3e71
ARG B12X_PACKAGE_TREE=95fdcb1cfea380480b8882fa44055cfef358ddbb
ARG LMCACHE_SOURCE_REVISION=822330db1160d2847538d53f9e6f4eeee2404f4d
ARG LMCACHE_SOURCE_TREE=008ac3e09ae5917aa0849147480d7bd5b9f8b37a
ARG LMCACHE_PACKAGE_TREE=fe5442fbf258accaa7f26d2bbb00d8b7b5c349ca
ARG LMCACHE_VERSION=0.5.5.dev0+glm53cache.engine.822330db
ARG FLASHKDA_SOURCE_REVISION=3b225bf26bb8e218928a1fe14751cb48cf31d11b
ARG FLASHKDA_EXTENSION_SHA256=16aece5ffb83c2dfb0355758bbbc9d6e0ea50a2cfc36ecee4936607d445aba0a
ARG SOURCE_LOCK_SHA256=e964f1c53c693eb83c3476c89a4a40e823f6d012f5ef9588ff5ae025ab18dd91
ARG BASE_LAUNCHER_SHA256=e149d469b999181ca626c450c12f531721f21c0fb697690f80c7617f591ea7b4
ARG RUNTIME_LAUNCHER_SHA256=69b0bf28564d5f497d6da1d6eb34ba45eb505be788247e8acce793bd88497d69
ARG LMCACHE_LAUNCHER_SHA256=c841af83547ecd6fd4bcf04cd8933b9f894eb5c9e27421130938b1168c0ac0c1
ARG CACHE_COMPLETE_LMCACHE_LAUNCHER_SHA256=a618a5c6e60e0b2d145cbf6a269cddb7fb032abd45d5b6ddc8d769150da15d02
ARG CACHE_MODE_LAUNCHER_SHA256=0c35ba35e24e5915b8da1ea2e1cae19e84ed1156057dd059800150d33f41ff18
ARG LMCACHE_RUNTIME_REQUIREMENTS_SHA256=7ac402540e2ba3f02fc1a443934ef3ff6e2e10c8ff81ae989e3958cfff38adf7
ARG QUALIFICATION_STATUS=qualified

# Install complete source-locked packages and launchers in one filesystem
# layer above the flattened CUDA/PyTorch/FlashInfer/InstantTensor foundation.
RUN --mount=type=bind,from=flashkda-artifact,source=/,target=/flashkda-artifacts,ro \
    --mount=type=bind,from=lmcache-build,source=/tmp/lmcache-artifacts,target=/lmcache-artifacts,ro \
    --mount=type=bind,from=vllm_source,source=/,target=/vllm-source,ro \
    --mount=type=bind,from=b12x_source,source=/,target=/b12x-source,ro \
    --mount=type=bind,source=serve-glm53-flash-nvfp4-dflash2.sh,target=/base-launcher,ro \
    --mount=type=bind,source=serve-glm53-flash-nvfp4-dflash2-scheduler-qos.sh,target=/runtime-launcher,ro \
    --mount=type=bind,source=serve-glm53-flash-lmcache.sh,target=/lmcache-launcher,ro \
    --mount=type=bind,source=serve-glm53-flash-lmcache-cache-complete.sh,target=/cache-complete-lmcache-launcher,ro \
    --mount=type=bind,source=serve-glm53-flash-cache-complete.sh,target=/cache-mode-launcher,ro \
    --mount=type=bind,source=glm53-r18-lmcache-runtime-requirements.txt,target=/lmcache-runtime-requirements,ro \
    --mount=type=bind,source=glm53-jovian-judgement-community-20260906-r27.source.lock,target=/source-lock,ro \
    test "$(sha256sum /source-lock | cut -d' ' -f1)" = "${SOURCE_LOCK_SHA256}" \
    && test "$(sha256sum /flashkda-artifacts/_flashkda_C.abi3.so | cut -d' ' -f1)" = \
      "${FLASHKDA_EXTENSION_SHA256}" \
    && grep -Fq "${FLASHKDA_SOURCE_REVISION}" \
      /vllm-source/cmake/external_projects/flashkda.cmake \
    && test "$(sha256sum /base-launcher | cut -d' ' -f1)" = "${BASE_LAUNCHER_SHA256}" \
    && test "$(sha256sum /runtime-launcher | cut -d' ' -f1)" = "${RUNTIME_LAUNCHER_SHA256}" \
    && test "$(sha256sum /lmcache-launcher | cut -d' ' -f1)" = "${LMCACHE_LAUNCHER_SHA256}" \
    && test "$(sha256sum /cache-complete-lmcache-launcher | cut -d' ' -f1)" = \
      "${CACHE_COMPLETE_LMCACHE_LAUNCHER_SHA256}" \
    && test "$(sha256sum /cache-mode-launcher | cut -d' ' -f1)" = "${CACHE_MODE_LAUNCHER_SHA256}" \
    && test "$(sha256sum /lmcache-runtime-requirements | cut -d' ' -f1)" = \
      "${LMCACHE_RUNTIME_REQUIREMENTS_SHA256}" \
    && grep -Fq 'def apply_top_k_top_p_probs' \
      /vllm-source/vllm/v1/sample/ops/topk_topp_sampler.py \
    && grep -Fq 'nvfp4_quantize' \
      /vllm-source/vllm/models/glm5next/nvidia/mtp_draft_head.py \
    && grep -Fq 'expandable_segments' \
      /vllm-source/vllm/config/vllm.py \
    && grep -Fq 'recurrent_instruction_boundary' \
      /vllm-source/vllm/renderers/online_renderer.py \
    && grep -Fq 'B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE' \
      /b12x-source/b12x/moe/fused_moe/_impl.py \
    && grep -Fq 'B12X_PCIE_ONESHOT_PDL' \
      /b12x-source/b12x/comm/pcie/_oneshot_cute.py \
    && grep -Fq 'B12X_MHC_PDL' \
      /b12x-source/b12x/norm/mhc/_kernels.py \
    && torch_before=$(/opt/venv/bin/python -c \
      'import torch; print(f"{torch.__version__}|{torch.version.cuda}|{int(torch._C._GLIBCXX_USE_CXX11_ABI)}")') \
    && test "${torch_before}" = '2.13.0|13.3|1' \
    && /opt/venv/bin/python -m pip install --no-cache-dir --no-deps \
      --requirement /lmcache-runtime-requirements \
    && /opt/venv/bin/python -m pip install --no-cache-dir --force-reinstall \
      --no-deps /lmcache-artifacts/*.whl \
    && torch_after=$(/opt/venv/bin/python -c \
      'import torch; print(f"{torch.__version__}|{torch.version.cuda}|{int(torch._C._GLIBCXX_USE_CXX11_ABI)}")') \
    && test "${torch_before}" = "${torch_after}" \
    && git -C /opt/glm53-flash/vllm ls-files -z -- vllm \
      | while IFS= read -r -d '' tracked_path; do \
          test -e "/vllm-source/${tracked_path}" \
            || rm -f "/opt/glm53-flash/vllm/${tracked_path}"; \
        done \
    && cp -a /vllm-source/vllm/. /opt/glm53-flash/vllm/vllm/ \
    && install -Dm755 /flashkda-artifacts/_flashkda_C.abi3.so \
      /opt/glm53-flash/vllm/vllm/_flashkda_C.abi3.so \
    && install -Dm755 /flashkda-artifacts/_flashkda_C.abi3.so \
      /opt/venv/lib/python3.12/site-packages/vllm/_flashkda_C.abi3.so \
    && test -s /opt/glm53-flash/vllm/vllm/_version.py \
    && test -s /opt/glm53-flash/vllm/vllm/vllm_flash_attn/layers/rotary.py \
    && test -s /opt/glm53-flash/vllm/vllm/third_party/triton_kernels/topk.py \
    && rm -rf /opt/glm53-flash/b12x/b12x \
    && cp -a /b12x-source/b12x /opt/glm53-flash/b12x/b12x \
    && find /opt/glm53-flash/vllm/vllm \
         /opt/glm53-flash/b12x/b12x \
         /opt/venv/lib/python3.12/site-packages/lmcache \
         -type d -name __pycache__ -prune -exec rm -r -- {} + \
    && find /opt/glm53-flash/vllm/vllm \
         /opt/glm53-flash/b12x/b12x \
         /opt/venv/lib/python3.12/site-packages/lmcache \
         -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && git -C /opt/glm53-flash/vllm add -A -- vllm \
    && git -C /opt/glm53-flash/b12x add -A -- b12x \
    && vllm_runtime_tree="$(git -C /opt/glm53-flash/vllm write-tree)" \
    && b12x_runtime_tree="$(git -C /opt/glm53-flash/b12x write-tree)" \
    && test "$(git -C /opt/glm53-flash/vllm rev-parse "${vllm_runtime_tree}:vllm")" = \
      "${VLLM_PACKAGE_TREE}" \
    && test "$(git -C /opt/glm53-flash/b12x rev-parse "${b12x_runtime_tree}:b12x")" = \
      "${B12X_PACKAGE_TREE}" \
    && install -Dm755 /lmcache-artifacts/liblmcache_cumem_shareable.so \
      /opt/lmcache/lib/liblmcache_cumem_shareable.so \
    && mkdir -p /opt/lmcache/artifacts \
    && cp /lmcache-artifacts/*.whl /opt/lmcache/artifacts/ \
    && install -Dm755 /lmcache-artifacts/liblmcache_cumem_shareable.so \
      /opt/lmcache/artifacts/liblmcache_cumem_shareable.so \
    && install -Dm644 /lmcache-artifacts/SHA256SUMS \
      /opt/lmcache/artifacts/SHA256SUMS \
    && install -Dm644 /lmcache-runtime-requirements \
      /opt/glm53-flash/lmcache-runtime-requirements.txt \
    && install -Dm755 /base-launcher \
      /usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh \
    && install -Dm755 /runtime-launcher \
      /usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh \
    && install -Dm755 /lmcache-launcher \
      /usr/local/bin/serve-glm53-flash-lmcache.sh \
    && install -Dm755 /cache-complete-lmcache-launcher \
      /usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh \
    && install -Dm755 /cache-mode-launcher \
      /usr/local/bin/serve-glm53-flash.sh \
    && install -Dm644 /source-lock /opt/glm53-flash/source.lock \
    && /opt/venv/bin/python -c \
      'import importlib.metadata as m; assert m.version("lmcache") == "'"${LMCACHE_VERSION}"'"' \
    && /opt/venv/bin/python -c \
      'import lmcache, lmcache.cuda_ops, lmcache.lmcache_native, lmcache.lmcache_fs; print(lmcache.__version__)' \
    && PYTHONPATH=/opt/glm53-flash/vllm:/opt/glm53-flash/b12x \
      /opt/venv/bin/python -c \
      'import vllm, vllm.vllm_flash_attn.layers.rotary, vllm.third_party.triton_kernels.topk, b12x' \
    && /opt/venv/bin/python -c \
      'from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import _bounded_relative_path_to_object_key, _object_key_to_bounded_relative_path; from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import _scan_existing_key_sizes; from lmcache.v1.multiprocess.transfer_context.async_engine_driven import AsyncEngineDrivenTransferContext; from lmcache.v1.multiprocess.server import _available_configured_l1_shm_bytes, _available_l1_shm_bytes_after_cleanup; print(AsyncEngineDrivenTransferContext, _available_configured_l1_shm_bytes, _available_l1_shm_bytes_after_cleanup, _object_key_to_bounded_relative_path, _bounded_relative_path_to_object_key, _scan_existing_key_sizes)' \
    && grep -Fq 'self._transfer_workspace_slot_count = self._commit_workers' \
      /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/transfer_context/async_engine_driven.py \
    && grep -Fq '_available_l1_shm_bytes_after_cleanup' \
      /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/server.py \
    && grep -Fq '_available_configured_l1_shm_bytes' \
      /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/server.py \
    && grep -Fq '_object_key_to_bounded_relative_path' \
      /opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py \
    && grep -Fq 'Failed to inspect native FS bounded cache root' \
      /opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/fs_native_l2_adapter.py \
    && test -s /opt/lmcache/lib/liblmcache_cumem_shareable.so \
    && nm -D /opt/lmcache/lib/liblmcache_cumem_shareable.so \
      | awk '$3 == "cuMemCreate" { found=1 } END { exit !found }' \
    && /opt/venv/bin/python -m pip check \
    && /opt/venv/bin/python -m compileall -q \
      /opt/glm53-flash/vllm/vllm \
      /opt/glm53-flash/b12x/b12x \
      /opt/venv/lib/python3.12/site-packages/lmcache \
    && /opt/venv/bin/python -m pip freeze --all | LC_ALL=C sort \
      > /opt/lmcache/python-packages.txt

ENV LMCACHE_EXPECTED_SOURCE_ROOT=/opt/venv/lib/python3.12/site-packages/lmcache \
    LOCAL_INFERENCE_CACHE_FINGERPRINT=cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096 \
    XDG_CACHE_HOME=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096 \
    VLLM_CACHE_ROOT=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/vllm \
    VLLM_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/vllm \
    TRITON_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/torchinductor \
    CUTE_DSL_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/cute-dsl \
    B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/b12x/cute \
    B12X_COMPILE_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/b12x/compile \
    SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/b12x/compile \
    CUDA_CACHE_PATH=/cache/jit/cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096/cuda \
    VLLM_GLM53_L2_PREFETCH=1 \
    VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0 \
    VLLM_GLM53_DFLASH_ATTN=1 \
    VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
    VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 \
    VLLM_GLM53_MTP_DRAFT_HEAD=nvfp4 \
    VLLM_MXFP8_LM_HEAD=0 \
    VLLM_MTP_NVFP4_LM_HEAD=0 \
    VLLM_LM_HEAD_A16=0 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=768KB \
    VLLM_GLM53_ONLINE_DENSE_MXFP8=0 \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
    VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
    B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 \
    B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
    B12X_DYNAMIC_SPLIT_LOW_SMEM=1 \
    B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
    B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
    B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
    B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 \
    B12X_PCIE_ONESHOT_THREADS=512 \
    B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
    B12X_PCIE_ONESHOT_PDL=1 \
    B12X_MHC_PDL=1 \
    NCCL_MIN_NCHANNELS=16 \
    NCCL_MAX_NCHANNELS=16 \
    NCCL_BUFFSIZE=2097152 \
    MAX_NUM_BATCHED_TOKENS=4096 \
    MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
    CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16 32 40 48 64 96 128 192 256" \
    PREFILL_SCHEDULE_INTERVAL=1 \
    FAIRNESS_ENGINE=compute_share \
    PREFILL_COMPUTE_SHARE=0.4 \
    GLM53_KDA_PREFILL_BACKEND=flashkda

LABEL local-inference.status="${QUALIFICATION_STATUS}" \
      local-inference.release.name="jovian-judgement-community-20260906-r27" \
      local-inference.release.overlay2-rootfs-layers="2" \
      local-inference.release.rootfs-format="flattened-runtime-plus-single-source-locked-package-layer" \
      local-inference.docker.commit="source-locked build; see org.opencontainers.image.source" \
      local-inference.runtime.base-image="voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43" \
      local-inference.runtime.launcher.sha256="${CACHE_MODE_LAUNCHER_SHA256}" \
      local-inference.runtime.source-lock.path="/opt/glm53-flash/source.lock" \
      local-inference.runtime.source-lock.sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.runtime.cache-fingerprint="cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096" \
      local-inference.cache.fingerprint="cu133-torch213-glm53-vllma336313d-fkda3b225bf-b12x95fdcb1c-lmcachefe5442fb-p2048-bt4096" \
      local-inference.cache.target-block-size="2048 for GPU-only serving; LMCache derives a DCP-aligned per-rank page" \
      local-inference.cache.recurrent-block-size="equal to the resolved target page" \
      local-inference.vllm.base-revision="7d66922a7bf0c9c7efe9a35a87df128f6b24e762" \
      local-inference.vllm.branch="integration/glm53-r27-release-20260906" \
      local-inference.vllm.foundation.commit="7d66922a7bf0c9c7efe9a35a87df128f6b24e762" \
      local-inference.vllm.commit="${VLLM_SOURCE_REVISION}" \
      local-inference.vllm.source-revision="${VLLM_SOURCE_REVISION}" \
      local-inference.vllm.upstream-base="7d66922a7bf0c9c7efe9a35a87df128f6b24e762" \
      local-inference.vllm.upstream-revision="7d66922a7bf0c9c7efe9a35a87df128f6b24e762" \
      local-inference.vllm.composition="dev/jovian-judgement plus source-locked GLM serving changes listed in /opt/glm53-flash/source.lock" \
      local-inference.vllm.integration-prs="#619,#573,#571,#572,#665,#552,#558,#553,#561,#664,#599,#666,#653" \
      local-inference.vllm.integration.prs="#619,#573,#571,#572,#665,#552,#558,#553,#561,#664,#599,#666,#653" \
      local-inference.vllm.integration.lock_sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.vllm.integration.tree="${VLLM_SOURCE_TREE}" \
      local-inference.vllm.merge-heads="${VLLM_SOURCE_REVISION}" \
      local-inference.vllm.patch_sha256="8e99d9eb4507132bba7640f17cc2046971f4ea8343dbb1dfc316029f4ff66d27" \
      local-inference.vllm.tree="${VLLM_SOURCE_TREE}" \
      local-inference.vllm.package-tree="${VLLM_PACKAGE_TREE}" \
      local-inference.runtime.vllm.composition.commit="${VLLM_SOURCE_REVISION}" \
      local-inference.runtime.vllm.port.commit="${VLLM_SOURCE_REVISION}" \
      local-inference.b12x.base-revision="a1bbd02781c7505754e7aa58a959c1a77891c690" \
      local-inference.b12x.branch="integration/glm53-r27-release-20260906" \
      local-inference.b12x.foundation.commit="a1bbd02781c7505754e7aa58a959c1a77891c690" \
      local-inference.b12x.commit="${B12X_SOURCE_REVISION}" \
      local-inference.b12x.source-revision="${B12X_SOURCE_REVISION}" \
      local-inference.b12x.upstream-base="a1bbd02781c7505754e7aa58a959c1a77891c690" \
      local-inference.b12x.upstream-revision="a1bbd02781c7505754e7aa58a959c1a77891c690" \
      local-inference.b12x.composition="B12X master plus source-locked GLM serving changes listed in /opt/glm53-flash/source.lock" \
      local-inference.b12x.integration-prs="#316,#267,#268,#269,#270,#272,#273,#278,#279,#280,#317" \
      local-inference.b12x.integration.prs="#316,#267,#268,#269,#270,#272,#273,#278,#279,#280,#317" \
      local-inference.b12x.integration.lock_sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.b12x.integration.tree="${B12X_SOURCE_TREE}" \
      local-inference.b12x.merge-heads="${B12X_SOURCE_REVISION}" \
      local-inference.b12x.patch_sha256="f1c91506e328a7cb2df19d368499677bc52135b510df8ad1fd95bd29020aeb04" \
      local-inference.b12x.tree="${B12X_SOURCE_TREE}" \
      local-inference.b12x.package-tree="${B12X_PACKAGE_TREE}" \
      local-inference.lmcache.ref="integration/glm53-r26-current-head-20260905" \
      local-inference.lmcache.commit="${LMCACHE_SOURCE_REVISION}" \
      local-inference.lmcache.integration.prs="#49,#50,#51" \
      local-inference.lmcache.integration.lock_sha256="${SOURCE_LOCK_SHA256}" \
      local-inference.lmcache.integration.tree="${LMCACHE_SOURCE_TREE}" \
      local-inference.lmcache.merge-heads="5c57de29557d8c0ff7bcbd1ae35c222608186d0b,94c5514a26e5eca22296995235f0e0df6b4f458d,63919a2c6c310f9b34de7049b9b28b77fab13ca0" \
      local-inference.lmcache.patch_sha256="4413fab809e48b4493b73367e8ca28c64a70412fea1bb40d1e7438c68c2c6fba" \
      local-inference.lmcache.upstream-base="7ed4675404a31f4ffafd98975899dc83832ba965" \
      local-inference.lmcache.tree="${LMCACHE_SOURCE_TREE}" \
      local-inference.lmcache.package-tree="${LMCACHE_PACKAGE_TREE}" \
      local-inference.lmcache.version="${LMCACHE_VERSION}" \
      local-inference.lmcache.transfer="GPU-isolated asynchronous engine-driven pinned shared memory" \
      local-inference.model.repository="local-inference-lab/GLM-5.3-Flash-NVFP4" \
      local-inference.model.revision="46aaae8a82032f77100f2f03e9cc11b391df3b4d" \
      local-inference.target.revision="46aaae8a82032f77100f2f03e9cc11b391df3b4d" \
      local-inference.draft.model="local-inference-lab/GLM-5.3-Flash-DFlash2" \
      local-inference.draft.revision="huggingface-main" \
      local-inference.draft.source-revision="external checkpoint; not embedded" \
      local-inference.draft.update-policy="huggingface-main" \
      local-inference.draft.quantization="MXFP8" \
      local-inference.draft.manifest.sha256="external checkpoint; not embedded" \
      local-inference.draft.weights.sha256="external checkpoint; not embedded" \
      local-inference.glm53.mtp-backends="attention:B12X,moe:Marlin" \
      local-inference.scheduler.max-num-batched-tokens="4096" \
      local-inference.scheduler.fairness-engine="compute_share" \
      local-inference.scheduler.prefill-compute-share="0.4" \
      local-inference.scheduler.prefill-schedule-interval="1" \
      local-inference.runtime.mtp-draft-head="NVFP4 draft-only vocabulary projection; BF16 target verifier" \
      local-inference.runtime.nccl="16 channels with a 2097152-byte buffer" \
      local-inference.qualification.prefill="llm-decode-bench 32K cell with 32314-32316 measured prompt tokens" \
      org.opencontainers.image.title="Jovian Judgement community GLM-5.3 cache-mode runtime" \
      org.opencontainers.image.description="GLM-5.3-Flash runtime with accelerated GPU-isolated engine-driven LMCache DRAM and filesystem tiers" \
      org.opencontainers.image.revision="${VLLM_SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md" \
      org.opencontainers.image.version="20260906-r27"

ENTRYPOINT ["/usr/local/bin/serve-glm53-flash.sh"]
