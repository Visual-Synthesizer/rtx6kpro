# DeepSeek-V4-Flash-DSpark v7

This page documents the v7 DS4 DSpark validation image. Unlike v6, this page is
for the DSpark checkpoint only:

```text
deepseek-ai/DeepSeek-V4-Flash-DSpark
```

DSpark is not regular MTP. The checkpoint stores draft weights under `mtp.*`,
but vLLM must launch it with `method=dspark`. The native validated draft block
size is `5`; larger speculative-token counts are not useful for this checkpoint.

## Docker Image

```text
voipmonitor/vllm:ds4dspark-v7-vea45be3-pr60-b12x284a2ea-cu132-20260628
voipmonitor/vllm@sha256:aabf42ab09c4d6f76fe633896604db29a478fcfacc435ae12d652f638f38e88b
```

Local image ID from the validated host:

```text
sha256:f2d3ccc20d84390538a7687fb3d83c4a563f2f7478c7991842305c6c217aa3b7
```

Runtime version:

```text
0.11.2.dev279+ds4dspark.v7.ea45be3.b12x284a2ea.fi25dd814.cu132.20260628
```

Build recipe:

```text
https://github.com/local-inference-lab/blackwell-llm-docker
build-ds4dspark-v7-cu132.sh
blackwell-llm-docker commit d0f73cf
```

vLLM source:

```text
https://github.com/local-inference-lab/vllm
branch codex/ds4dspark-v7-pr60-20260628
commit ea45be31552df833b956d24b4eb814829c5ec68d
```

This image includes:

| Component | Commit / PR | Notes |
|---|---|---|
| DSpark vLLM work | `ea45be315` | Adds DeepSeek V4 DSpark speculative decoding and B12X DSpark WO projection fix. |
| Internal DSpark PR | `local-inference-lab/vllm#61` | `[dspark] Add DeepSeek V4 DSpark speculative decoding`. |
| PR60 | `56fb5d890` | SM120 FlashInfer sparse MLA DCP support, included in the image stack. |
| B12X | `284a2eae` | Same B12X stack used by the Eldritch line. |
| FlashInfer | `25dd814e` | Built for CUDA 13.2 / SM120. |

## Model

```text
deepseek-ai/DeepSeek-V4-Flash-DSpark
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7
```

Important DSpark checkpoint settings:

```text
dspark_block_size=5
dspark_target_layer_ids=[40,41,42]
n_mtp_layers=3
dspark_noise_token_id=128799
dspark_markov_rank=256
```

## Variants

| Variant | Attention | MoE / linear | Validated role |
|---|---|---|---|
| B12X | `B12X_MLA_SPARSE` | `b12x` MoE + `b12x` linear | Stable after the v7 B12X WO fix; useful for cc1 DSpark only in this matrix. |
| Lucifer CUTLASS | `FLASHINFER_MLA_SPARSE_DSV4` | `flashinfer_cutlass` MXFP4 MoE | Best DSpark decode profile in the v7 matrix. |
| Lucifer default | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path | Good DSpark decode, slightly better DSpark-on long prefill than CUTLASS. |

## Docker Compose

Set `VARIANT` to `b12x`, `lucifer-cutlass`, or `lucifer-default`.
Set `DSPARK_TOKENS=5` for DSpark on, or `DSPARK_TOKENS=0` for DSpark off.

Validated defaults for DSpark on are `GRAPH_CAP=512`,
`GPU_MEMORY_UTILIZATION=0.92`, and `DSPARK_TOKENS=5`. For DSpark off use
`GRAPH_CAP=256` and `GPU_MEMORY_UTILIZATION=0.90`.

TileLang ignores `XDG_CACHE_HOME`, so keep its cache on the persistent `/cache`
mount explicitly. Without these two environment variables, kernels such as
`mhc_pre_big_fuse_with_norm_tilelang` are cached only under
`/root/.tilelang/cache` inside the container layer and are recompiled after a
new `docker run` or compose recreate:

```text
TILELANG_CACHE_DIR=/cache/tilelang
TILELANG_TMP_DIR=/cache/tilelang/tmp
```

```yaml
services:
  ds4dspark:
    image: ${IMAGE:-voipmonitor/vllm:ds4dspark-v7-vea45be3-pr60-b12x284a2ea-cu132-20260628}
    container_name: ${NAME:-ds4dspark-v7}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    init: true
    ulimits:
      memlock: -1
      nofile: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface:ro
      - ${CACHE_ROOT:-/root/.cache/vllm-ds4dspark-v7}:/cache
      - ${TMP_ROOT:-/root/.cache/vllm-ds4dspark-v7-tmp}:/container-tmp
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      SAFETENSORS_FAST_GPU: "1"
      TMPDIR: /container-tmp
      XDG_CACHE_HOME: /cache
      VLLM_CACHE_DIR: /cache/vllm
      TRITON_CACHE_DIR: /cache/triton
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      TORCH_EXTENSIONS_DIR: /cache/torch_extensions
      FLASHINFER_WORKSPACE_BASE: /cache/flashinfer
      TILELANG_CACHE_DIR: /cache/tilelang
      TILELANG_TMP_DIR: /cache/tilelang/tmp
      MODEL_PATH: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7}
      PORT: ${PORT:-8000}
      VARIANT: ${VARIANT:-lucifer-cutlass}
      DSPARK_TOKENS: ${DSPARK_TOKENS:-5}
      DSPARK_SAMPLE: ${DSPARK_SAMPLE:-probabilistic}
      TP_SIZE: ${TP_SIZE:-2}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-128}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.92}
      GRAPH_CAP: ${GRAPH_CAP:-512}
      PREFIX_CACHE: ${PREFIX_CACHE:-1}
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

        EXTRA_ARGS=()
        case "$${VARIANT}" in
          b12x)
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-4096}"
            export VLLM_USE_B12X_WO_PROJECTION=1
            export VLLM_USE_B12X_MHC=1
            export VLLM_USE_B12X_FP8_GEMM=1
            export VLLM_USE_B12X_MOE=1
            export VLLM_USE_B12X_SPARSE_INDEXER=1
            export VLLM_ENABLE_PCIE_ALLREDUCE=1
            export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
            export B12X_MLA_SM120_UNIFIED=1
            export B12X_MHC_MAX_TOKENS=16384
            export B12X_DENSE_SPLITK_TURBO=1
            export B12X_W4A16_TC_DECODE=1
            EXTRA_ARGS=(--attention-backend B12X_MLA_SPARSE --moe-backend b12x --linear-backend b12x)
            ;;
          lucifer-cutlass)
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-8192}"
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            EXTRA_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass --disable-custom-all-reduce)
            ;;
          lucifer-default)
            MAX_NUM_BATCHED_TOKENS="$${MAX_NUM_BATCHED_TOKENS:-8192}"
            export VLLM_ENABLE_PCIE_ALLREDUCE=0
            export VLLM_PCIE_ALLREDUCE_BACKEND=cpp
            EXTRA_ARGS=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4 --disable-custom-all-reduce)
            ;;
          *)
            echo "Unknown VARIANT=$${VARIANT}" >&2
            exit 2
            ;;
        esac

        SPEC_ARGS=()
        if [ "$${DSPARK_TOKENS}" != "0" ]; then
          SPEC_ARGS=(--speculative-config "{\"model\":\"$${MODEL_PATH}\",\"method\":\"dspark\",\"num_speculative_tokens\":$${DSPARK_TOKENS},\"draft_sample_method\":\"$${DSPARK_SAMPLE}\"}")
        fi

        PREFIX_ARGS=(--enable-prefix-caching)
        if [ "$${PREFIX_CACHE}" != "1" ]; then
          PREFIX_ARGS=(--no-enable-prefix-caching)
        fi

        exec vllm serve "$${MODEL_PATH}" \
          --served-model-name DeepSeek-V4-Flash-DSpark \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --kv-cache-dtype fp8 \
          --block-size 256 \
          --load-format auto \
          --tensor-parallel-size "$${TP_SIZE}" \
          --decode-context-parallel-size 1 \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-model-len 262144 \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-cudagraph-capture-size "$${GRAPH_CAP}" \
          --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
          --async-scheduling \
          --no-scheduler-reserve-full-isl \
          --enable-chunked-prefill \
          --enable-flashinfer-autotune \
          --tokenizer-mode deepseek_v4 \
          --tool-call-parser deepseek_v4 \
          --reasoning-parser deepseek_v4 \
          --enable-auto-tool-choice \
          --default-chat-template-kwargs.thinking=true \
          --default-chat-template-kwargs.reasoning_effort=high \
          "$${SPEC_ARGS[@]}" \
          "$${EXTRA_ARGS[@]}" \
          "$${PREFIX_ARGS[@]}"
```

## Single Docker Run

Exact validated-style launch for Lucifer CUTLASS with DSpark on:

```bash
docker rm -f ds4dspark-v7 2>/dev/null || true

docker run -d --name ds4dspark-v7 \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host --init \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /root/.cache/vllm-ds4dspark-v7:/cache \
  -v /root/.cache/vllm-ds4dspark-v7-tmp:/container-tmp \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e TMPDIR=/container-tmp \
  -e XDG_CACHE_HOME=/cache \
  -e VLLM_CACHE_DIR=/cache/vllm \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  voipmonitor/vllm:ds4dspark-v7-vea45be3-pr60-b12x284a2ea-cu132-20260628 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7 --served-model-name DeepSeek-V4-Flash-DSpark --host 0.0.0.0 --port 8000 --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --load-format auto --tensor-parallel-size 2 --decode-context-parallel-size 1 --gpu-memory-utilization 0.92 --max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 512 --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" --async-scheduling --no-scheduler-reserve-full-isl --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 --enable-auto-tool-choice --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high --speculative-config "{\"model\":\"/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7\",\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\"}" --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --kernel-config.moe_backend flashinfer_cutlass --disable-custom-all-reduce'
```

## Benchmarks

All rows below are TP2 on RTX PRO 6000 Blackwell, GPUs `0,1`,
`kv-cache-dtype=fp8`, `max-model-len=262144`, `max-num-seqs=128`,
prefix cache enabled, clean image with `OVERLAY=0`, and no runtime source patch.

Decode cells are 30 second sustained decode at `ctx=0`. B12X uses
`MAX_NUM_BATCHED_TOKENS=4096`; Lucifer variants use `8192`.

### Decode Throughput

Aggregate decode tok/s:

| Variant | DSpark | cc1 | cc64 | cc128 |
|---|---:|---:|---:|---:|
| B12X | off | 130.6 | 1384.2 | 1887.4 |
| B12X | on | 197.9 | 954.2 | 1168.6 |
| Lucifer default | off | 116.0 | 1788.6 | 2960.0 |
| Lucifer default | on | 189.7 | 2316.2 | 3195.4 |
| Lucifer CUTLASS | off | 122.7 | 1973.5 | 3236.3 |
| Lucifer CUTLASS | on | 208.2 | 2486.2 | 3333.2 |

DSpark on/off decode deltas:

| Variant | cc1 | cc64 | cc128 |
|---|---:|---:|---:|
| B12X | +51.5% | -31.1% | -38.1% |
| Lucifer default | +63.6% | +29.5% | +8.0% |
| Lucifer CUTLASS | +69.7% | +26.0% | +3.0% |

Lucifer CUTLASS + DSpark is the best v7 decode profile in this matrix. B12X +
DSpark is stable, but it is not a good high-concurrency DSpark profile yet.

### Prefill Throughput

Client-side prompt tokens / TTFT:

| Variant | DSpark | 8k tok/s | 64k tok/s | 128k tok/s |
|---|---:|---:|---:|---:|
| B12X | off | 7589 | 5429 | 4061 |
| B12X | on | 7311 | 5346 | 4025 |
| Lucifer default | off | 13318 | 12668 | 11659 |
| Lucifer default | on | 12592 | 12505 | 11510 |
| Lucifer CUTLASS | off | 13021 | 12369 | 11389 |
| Lucifer CUTLASS | on | 12627 | 12341 | 11359 |

DSpark is a decode feature; prefill should be read mainly as a regression check.

### DSpark Acceptance

Final Prometheus counters after the full matrix DSpark-on cases:

| Variant | Drafts | Accepted tokens | Draft tokens | Accepted / draft | Draft-token acceptance |
|---|---:|---:|---:|---:|---:|
| B12X | 30006 | 53183 | 150030 | 1.77 | 35.45% |
| Lucifer default | 75047 | 130202 | 375235 | 1.73 | 34.70% |
| Lucifer CUTLASS | 78993 | 136730 | 394965 | 1.73 | 34.62% |

Short repeated-prompt prefix-cache smoke on the same final image showed higher
acceptance for that prompt: `183 / 365` draft tokens, or `50.1%`.

## Prefix Cache

Prefix cache is enabled and works with DSpark on the final v7 image. DSpark uses
safe conservative semantics: after a prefix-cache hit, draft proposals are
deferred until DSpark's private rolling KV window is rebuilt from fresh target
rows.

Final-image prefix-cache smoke:

| Request | Prompt tokens | Completion tokens | Elapsed | Finish |
|---|---:|---:|---:|---|
| First identical prompt | 37816 | 127 | 3.433s | `stop` |
| Second identical prompt | 37816 | 128 | 0.604s | `length` |

Final metrics after the two requests:

```text
prefix_cache_queries_total=75632
prefix_cache_hits_total=37632
prompt_tokens_cached_total=37632
spec_decode_num_drafts_total=73
spec_decode_num_draft_tokens_total=365
spec_decode_num_accepted_tokens_total=183
```

The full benchmark matrix also launched every row with prefix cache enabled, but
those benchmark prompts did not repeat, so the matrix intentionally has zero
prefix-cache hits.

## Validation

Clean-image checks completed before publishing this page:

```text
ruff check vllm/models/deepseek_v4/nvidia/dspark.py tests/v1/spec_decode/test_dspark.py
pytest -q tests/v1/spec_decode/test_dspark.py  # 10 passed
git diff --check
```

Runtime validation:

| Check | Result |
|---|---|
| Docker import/version smoke | Passed, final v7 version reported. |
| B12X DSpark clean-image smoke | Passed after fused WO projection fix. |
| Full B12X/Lucifer x DSpark off/on matrix | Passed, all six cases completed. |
| Target CUDA graphs | PIECEWISE and FULL captured. |
| DSpark CUDA graphs | FULL block-forward graph captured. |
| Prefix cache repeated-prompt smoke | Passed on final pushed image. |

## Artifacts

```text
/root/bench-results/ds4dspark-v7-full-ea45be3-20260628/
/root/bench-results/ds4dspark-v7-full-ea45be3-20260628/full-matrix.log
/root/bench-results/ds4dspark-v7-prefix-ea45be3-20260628/
/root/bench-results/ds4dspark-v7-clean-smoke-20260628/b12x-on-clean/
/root/vllm/blackwell-llm-docker/logs/build-ds4dspark-v7-vea45be3-20260628.log
```

Benchmark runner:

```text
/root/vllm/research/dspark-20260627/run-ds4dspark-v7-bench.sh
```

Full matrix command:

```bash
IMAGE=voipmonitor/vllm:ds4dspark-v7-vea45be3-pr60-b12x284a2ea-cu132-20260628 \
GPUS=0,1 \
TP=2 \
PORT=5960 \
OUT=/root/bench-results/ds4dspark-v7-full-ea45be3-20260628 \
/root/vllm/research/dspark-20260627/run-ds4dspark-v7-bench.sh
```

## Caveats

- v7 is DSpark-checkpoint only. Use v6 for the standard DeepSeek-V4-Flash MTP
  sweep.
- `DSPARK_TOKENS=5` is the validated native setting. Treat values above 5 as
  unsupported for this checkpoint.
- B12X + DSpark is stable, but high-concurrency decode regresses versus B12X
  DSpark off in the current implementation.
- Prefix cache is enabled and validated, but DSpark draft private-cache
  rehydration from cached target KV is conservative rather than zero-wait.
- The internal DSpark PR is ready for local review, but an official upstream
  vLLM PR still needs a clean split from local B12X/Lucifer-only dependencies.
