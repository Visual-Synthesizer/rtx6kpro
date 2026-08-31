# Kimi-K2.6 v6 on 8x RTX PRO 6000 Blackwell

Status: measured 2026-05-26. This page documents the Kimi-K2.6 v6 runtime
profile using the GLM 5.1 v5 upstream-main vLLM image and the new LightSeek
Kimi-K2.6 Eagle3.1 MLA draft model.

This page records the exact measured configuration. The LightSeek model card
targets `nvidia/Kimi-K2.6-NVFP4`; the benchmark below used our existing local
Kimi v5 target checkpoint because that is the target currently present on the
8x RTX PRO 6000 host.

## Docker Compose

Create the compose file:

```bash
cat >/tmp/kimi-k26-v6.compose.yaml <<'EOF'
services:
  kimi-k26-v6:
    image: ${IMAGE:-voipmonitor/vllm:glm51-v5-upstreammain-vllm4cdbe04-b12xf6abdd2-flashinfer56d537a-20260526}
    container_name: ${NAME:-kimi-k26-v6}
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail
      unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

      spec_args=()
      if [[ "$${KIMI_DISABLE_MTP:-0}" != "1" ]]; then
        spec_args+=(--speculative-config "$${KIMI_SPEC_CONFIG}")
      fi

      exec /opt/venv/bin/vllm serve "$${MODEL}" \
        --served-model-name Kimi-K2.6 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 1 \
        --decode-context-parallel-size "$${DCP_SIZE}" \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --async-scheduling \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len 262144 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 128 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode weights \
        --attention-backend TRITON_MLA \
        --kv-cache-dtype fp8 \
        --enable-flashinfer-autotune \
        --max-cudagraph-capture-size 512 \
        --reasoning-parser kimi_k2 \
        --tool-call-parser kimi_k2 \
        --enable-auto-tool-choice \
        "$${spec_args[@]}"
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v6}/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v6}/jit:/cache/jit
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v6}/triton:/root/.cache/triton
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v6}/torchinductor:/root/.cache/torchinductor
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v6}/vllm:/root/.cache/vllm
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      NVIDIA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      OMP_NUM_THREADS: "16"
      SAFETENSORS_FAST_GPU: "1"
      CUTE_DSL_ARCH: sm_120a
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      VLLM_USE_B12X_SPARSE_INDEXER: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      PORT: ${PORT:-8402}
      MODEL: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}
      TP_SIZE: "8"
      DCP_SIZE: ${DCP:-8}
      GPU_MEMORY_UTILIZATION: ${GPU_MEM:-0.90}
      MAX_MODEL_LEN: "262144"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "128"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "512"
      ATTENTION_BACKEND: TRITON_MLA
      KV_CACHE_DTYPE: fp8
      LOAD_FORMAT: fastsafetensors
      ENABLE_PREFIX_CACHING: "1"
      ENABLE_CHUNKED_PREFILL: "1"
      ENABLE_ASYNC_SCHEDULING: "1"
      KIMI_DISABLE_MTP: ${KIMI_DISABLE_MTP:-0}
      KIMI_SPEC_CONFIG: '{"model":"${DRAFT_MODEL_PATH:-/root/.cache/huggingface/hub/models--lightseekorg--kimi-k2.6-eagle3.1-mla/snapshots/35194ee8feb2826812f716eb42a924f99a5404f3}","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'
      HF_HOME: /root/.cache/huggingface
      HUGGINGFACE_HUB_CACHE: /root/.cache/huggingface/hub
      XDG_CACHE_HOME: /cache/jit
      CUDA_CACHE_PATH: /cache/jit
      VLLM_CACHE_DIR: /cache/jit/vllm
      TVM_FFI_CACHE_DIR: /cache/jit/tvm-ffi
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      VLLM_CACHE_ROOT: /root/.cache/vllm
      TRITON_CACHE_DIR: /root/.cache/triton
      TORCHINDUCTOR_CACHE_DIR: /root/.cache/torchinductor
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      CUTE_DSL_CACHE_DIR: /root/.cache/cutlass_dsl
EOF
```

Run the measured profile:

```bash
DCP=8 KIMI_DISABLE_MTP=0 GPU_MEM=0.90 docker compose -f /tmp/kimi-k26-v6.compose.yaml up -d
```

Stop it:

```bash
docker compose -f /tmp/kimi-k26-v6.compose.yaml down
```

## Image And Checkpoints

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:glm51-v5-upstreammain-vllm4cdbe04-b12xf6abdd2-flashinfer56d537a-20260526` |
| vLLM label | `4cdbe047c596342f2e924101798cd907469d0090` |
| B12X label | `f6abdd287994141712f8401645afcc3e4b25dbc8` |
| FlashInfer label | `56d537a106024eb25f4d4a186eadc226990a9185` |
| Target model | `/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58` |
| Draft model | `/root/.cache/huggingface/hub/models--lightseekorg--kimi-k2.6-eagle3.1-mla/snapshots/35194ee8feb2826812f716eb42a924f99a5404f3` |
| Served name | `Kimi-K2.6` |

The measured runtime uses `TRITON_MLA` for target and draft attention. The
`tokenspeed_mla` vLLM wrapper exists in this image, but the external
`tokenspeed_mla` package is not installed and this v6 run does not use it.
The logs show `Using AttentionBackendEnum.TRITON_MLA backend` and
`Using FLASH_ATTN MLA prefill backend`.

`VLLM_USE_B12X_SPARSE_INDEXER=0` is set explicitly because this image can carry
an empty inherited value for that env var. Leaving it empty makes newer vLLM
workers fail while parsing `int("")`. Kimi v6 uses `TRITON_MLA`, not the B12X
sparse MLA path.

Never set `NCCL_GRAPH_FILE=` as an empty env var in this profile. The compose
command unsets `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
`VLLM_B12X_MLA_EXTEND_MAX_CHUNKS`.

## Readiness Checks

```bash
curl -fsS http://127.0.0.1:8402/health
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k26-v6 2>&1 | rg 'Application startup complete|GPU KV cache size|Eagle3 auxiliary|TRITON_MLA|FLASH_ATTN MLA prefill'
```

Expected log markers from the measured run:

```text
Resolved architecture: KimiK25ForConditionalGeneration
Resolved architecture: Eagle3DeepseekV2ForCausalLM
Using AttentionBackendEnum.TRITON_MLA backend.
Using FLASH_ATTN MLA prefill backend.
Using Eagle3 auxiliary layers from model: (2, 30, 58)
GPU KV cache size: 2,725,376 tokens
Maximum concurrency for 262,144 tokens per request: 10.40x
Application startup complete.
```

Short smoke test:

```bash
curl -fsS http://127.0.0.1:8402/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K2.6",
    "messages": [{"role": "user", "content": "Reply with one word: ready"}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq '.choices[0].finish_reason, .choices[0].message.content, .choices[0].message.reasoning'
```

## Benchmark Command

Use `llm_decode_bench.py` v0.4.23 or newer. The DCP8 and DCP1 numbers below
were measured with scout prefill enabled and sustained decode enabled.
Burst/E2E decode was not run.

Measured JSON source on the local host:

```bash
/root/benchmark_results.json
```

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8402 \
  --model Kimi-K2.6 \
  --contexts 0,16k,32k,64k,128k \
  --concurrency 1,2,4,8,16,32,64,128 \
  --duration 30 \
  --dcp-size 8 \
  --output /root/bench-results/kimi-k26-v6-lightseek-eagle31-dcp8-mtp1-20260526/result.json
```

## DCP8 Primary Results

Configuration:

| Field | Value |
|---|---|
| Benchmark | `llm_decode_bench.py v0.4.23` |
| Result timestamp | `2026-05-26T15:48:42.796168` |
| Runtime | vLLM V2 model runner |
| DCP | 8 |
| MTP | Eagle3.1, `num_speculative_tokens=3` |
| GPU memory utilization | `0.90` |
| KV cache | fp8, `2,725,376` tokens |
| Attention | target `TRITON_MLA`, draft `TRITON_MLA`, prefill selected `FLASH_ATTN MLA` |
| Allreduce | PCIe custom allreduce disabled, PYNCCL fallback |
| Aggregate decode source | `openai_continuous_usage` |
| Raw result | captured from the previous DCP8 `/root/benchmark_results.json`; the current file was later overwritten by the DCP1 sweep |

### Prefill Speed

Scout prefill rows measure prompt tokens divided by TTFT. These scout requests
are part of the decode benchmark flow, so they are useful as a practical prefill
signal rather than a synthetic prefill-only microbenchmark.

| Context | Prompt tokens | TTFT s | Client tok/s | N |
|---:|---:|---:|---:|---:|
| 8k | 8,188 | 1.09 | 7,542 | 1 |
| 16k | 16,230 | 2.27 | 7,151 | 1 |
| 32k | 32,308 | 4.90 | 6,598 | 1 |
| 64k | 64,469 | 11.28 | 5,717 | 1 |
| 128k | 128,786 | 28.13 | 4,577 | 1 |

### Sustained Decode Aggregate tok/s

This is the main kernel/scheduler regression signal. `N/A` means the cell was
skipped because it did not fit in KV cache.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95.3 | 163.1 | 233.1 | 333.8 | 528.5 | 686.7 | 955.3 | 1159.5 |
| 16k | 94.9 | 141.7 | 214.9 | 311.7 | 448.5 | 574.1 | 762.0 | 912.8 |
| 32k | 91.8 | 135.1 | 197.8 | 282.6 | 390.7 | 488.3 | 628.0 | N/A |
| 64k | 80.9 | 121.5 | 172.0 | 222.9 | 298.9 | 376.7 | N/A | N/A |
| 128k | 70.3 | 97.6 | 138.0 | 174.2 | 211.0 | N/A | N/A | N/A |

### Speculative Acceptance Rate

Values are `server_spec_accept_rate` from the benchmark JSON. This is the main
Eagle3.1 acceptance signal for this run. `server_spec_accept_length` was present
but stayed `0.0` in this JSON, so use the rate matrix below.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.6% | 44.0% | 38.7% | 32.9% | 34.2% | 38.0% | 39.8% | 45.6% |
| 16k | 45.5% | 50.5% | 36.3% | 30.1% | 37.5% | 39.8% | 41.8% | 42.4% |
| 32k | 44.2% | 40.9% | 29.3% | 39.8% | 36.2% | 42.4% | 44.0% | N/A |
| 64k | 35.9% | 29.5% | 35.0% | 42.0% | 38.3% | 54.6% | N/A | N/A |
| 128k | 32.3% | 43.9% | 34.4% | 43.1% | 39.9% | N/A | N/A | N/A |

### Per-Request tok/s

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95.3 | 81.5 | 58.3 | 41.7 | 33.0 | 21.5 | 14.9 | 9.1 |
| 16k | 94.9 | 70.9 | 53.7 | 39.0 | 28.0 | 17.9 | 11.9 | 7.1 |
| 32k | 91.8 | 67.5 | 49.4 | 35.3 | 24.4 | 15.3 | 9.8 | N/A |
| 64k | 80.9 | 60.8 | 43.0 | 27.9 | 18.7 | 11.8 | N/A | N/A |
| 128k | 70.3 | 48.8 | 34.5 | 21.8 | 13.2 | N/A | N/A | N/A |

### Selected TTFT / ITL

Cells are `TTFT ms / ITL ms`. ITL is computed from observed generated tokens.

| ctx \ conc | 1 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|
| 0 | 62 / 10 | 186 / 29 | 469 / 44 | 735 / 63 | 880 / 102 |
| 16k | 236 / 10 | 2k / 35 | 4k / 54 | 5k / 83 | 7k / 146 |
| 32k | 377 / 11 | 4k / 39 | 8k / 63 | 8k / 102 | N/A |
| 64k | 680 / 12 | 7k / 51 | 13k / 83 | N/A | N/A |
| 128k | 1k / 14 | 14k / 71 | N/A | N/A | N/A |

## DCP8 Consolidated Findings

- Best single-request decode: `95.3 tok/s` at `ctx0`, `94.9 tok/s` at `16k`, `91.8 tok/s` at `32k`, `80.9 tok/s` at `64k`, and `70.3 tok/s` at `128k`.
- Best aggregate decode in this sweep: `1159.5 tok/s` at `ctx0/concurrency128`.
- Highest long-context measured aggregate: `211.0 tok/s` at `128k/concurrency16`; higher 128k concurrencies did not fit in KV cache.
- Prefill scout throughput declines from `7,542 tok/s` at 8k to `4,577 tok/s` at 128k.
- Speculative acceptance ranged from `24.6%` to `54.6%`; the simple mean across measured cells was `39.0%`, and the output-token-weighted mean was about `40.2%`.
- Best acceptance cell was `64k/concurrency32` at `54.6%`; lowest was `ctx0/concurrency1` at `24.6%`.
- The run is GPU-bound: sustained cells show about `97-100%` GPU utilization and `99.2%` VRAM occupancy.
- PCIe traffic is workload dependent. The largest sampled short-context cell was about `223 GB/s rx / 223 GB/s tx` at `ctx0/concurrency128`; long-context `128k/concurrency16` was about `42 GB/s rx / 41 GB/s tx`.
- Peak sampled board power in this run was about `2.76 kW` total across the 8 GPUs; max reported GPU temperature was `64C`.

## DCP1 Primary Results

The DCP1 run keeps the same image, target checkpoint, LightSeek Eagle3.1 MLA
draft, MTP settings, attention backend, fp8 KV cache, and `GPU_MEM=0.90`. Only
`DCP=1` was changed from the DCP8 profile.

Configuration:

| Field | Value |
|---|---|
| Benchmark | `llm_decode_bench.py v0.4.23` |
| Result timestamp | `2026-05-26T16:39:37.170417` |
| Runtime | vLLM V2 model runner |
| DCP | 1 |
| MTP | Eagle3.1, `num_speculative_tokens=3` |
| GPU memory utilization | `0.90` |
| KV cache | fp8, `340,672` tokens |
| Maximum concurrency for 262,144 tokens | `1.30x` |
| Attention | target `TRITON_MLA`, draft `TRITON_MLA`, prefill selected `FLASH_ATTN MLA` |
| Allreduce | PCIe custom allreduce disabled, PYNCCL fallback |
| Aggregate decode source | `openai_continuous_usage` |
| Raw result | `/root/benchmark_results.json` |

### DCP1 Prefill Speed

| Context | Prompt tokens | TTFT s | Client tok/s | N |
|---:|---:|---:|---:|---:|
| 8k | 8,186 | 1.09 | 7,528 | 1 |
| 16k | 16,228 | 2.26 | 7,192 | 1 |
| 32k | 32,306 | 4.82 | 6,701 | 1 |
| 64k | 64,467 | 10.92 | 5,906 | 1 |
| 128k | 128,784 | 26.58 | 4,845 | 1 |

### DCP1 Sustained Decode Aggregate tok/s

`N/A` means the cell was not measured in this DCP1 sweep because it did not fit
the available KV cache.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128.4 | 216.9 | 354.1 | 493.4 | 743.1 | 1104.4 | 1652.9 | 2377.8 |
| 16k | 111.3 | 186.8 | 259.8 | 366.3 | 491.4 | N/A | N/A | N/A |
| 32k | 94.4 | 162.0 | 220.9 | 270.4 | N/A | N/A | N/A | N/A |
| 64k | 82.1 | 119.8 | 158.0 | N/A | N/A | N/A | N/A | N/A |
| 128k | 60.6 | 86.3 | N/A | N/A | N/A | N/A | N/A | N/A |

### DCP1 Speculative Acceptance Rate

Values are `server_spec_accept_rate` from the benchmark JSON.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 41.7% | 42.2% | 43.3% | 56.7% | 34.0% | 34.4% | 38.4% | 38.5% |
| 16k | 32.1% | 33.0% | 52.2% | 51.7% | 45.1% | N/A | N/A | N/A |
| 32k | 42.0% | 44.4% | 35.6% | 33.9% | N/A | N/A | N/A | N/A |
| 64k | 42.3% | 32.6% | 31.0% | N/A | N/A | N/A | N/A | N/A |
| 128k | 21.2% | 32.5% | N/A | N/A | N/A | N/A | N/A | N/A |

### DCP1 Per-Request tok/s

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128.4 | 108.4 | 88.5 | 61.7 | 46.4 | 34.5 | 25.8 | 18.6 |
| 16k | 111.3 | 93.4 | 65.0 | 45.8 | 30.7 | N/A | N/A | N/A |
| 32k | 94.4 | 81.0 | 55.2 | 33.8 | N/A | N/A | N/A | N/A |
| 64k | 82.1 | 59.9 | 39.5 | N/A | N/A | N/A | N/A | N/A |
| 128k | 60.6 | 43.1 | N/A | N/A | N/A | N/A | N/A | N/A |

### DCP1 Selected TTFT / ITL

Cells are `TTFT ms / ITL ms`.

| ctx \ conc | 1 | 16 | 32 | 64 | 128 |
|---:|---:|---:|---:|---:|---:|
| 0 | 50 / 8 | 164 / 21 | 213 / 27 | 350 / 36 | 504 / 50 |
| 16k | 163 / 9 | 1.4k / 32 | N/A | N/A | N/A |
| 32k | 286 / 10 | N/A | N/A | N/A | N/A |
| 64k | 533 / 12 | N/A | N/A | N/A | N/A |
| 128k | 1k / 16 | N/A | N/A | N/A | N/A |

## DCP1 Consolidated Findings

- DCP1 is faster per measured decode cell than DCP8, but has much less KV capacity: `340,672` tokens versus DCP8 `2,725,376` tokens.
- Best single-request decode in DCP1: `128.4 tok/s` at `ctx0`, `111.3 tok/s` at `16k`, `94.4 tok/s` at `32k`, `82.1 tok/s` at `64k`, and `60.6 tok/s` at `128k`.
- Best aggregate decode in this DCP1 sweep: `2377.8 tok/s` at `ctx0/concurrency128`.
- Highest long-context measured aggregate: `86.3 tok/s` at `128k/concurrency2`; higher 128k concurrencies were not measured because the DCP1 KV budget does not fit them.
- Prefill scout throughput declines from `7,528 tok/s` at 8k to `4,845 tok/s` at 128k.
- Speculative acceptance ranged from `21.2%` to `56.7%`; the simple mean across measured cells was `39.0%`, and the output-token-weighted mean was about `39.5%`.
- The run is GPU-bound in measured cells: max sampled GPU utilization reached `99%`, max VRAM occupancy was `95.2%`, peak sampled board power was about `2.77 kW`, and max reported GPU temperature was `64C`.

## Notes

- This page currently records DCP8 + MTP on and DCP1 + MTP on. DCP2/4
  comparisons still need to be measured separately.
- The current target checkpoint is `moonshotai/Kimi-K2.6`. If switching to the
  LightSeek card's suggested `nvidia/Kimi-K2.6-NVFP4` target, treat that as a new
  profile and remeasure.
- `tokenspeed_mla` is not part of this profile. The image has vLLM wrapper code
  for it, but the external package is absent and the measured backend is
  `TRITON_MLA`.
- Keep persistent cache mounts. The first run compiles target, Eagle head, full
  CUDA graphs, Eagle prefill graphs, Eagle decode graphs, and FlashInfer runtime
  autotune buckets.
- `NCCL_GRAPH_FILE=` must not be exported as an empty env var. Leave it unset.
- Do not set `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` in this Kimi profile.
