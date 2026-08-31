# Kimi-K2.7-Code v2 on Eldritch

This page documents Kimi-K2.7-Code on the shared Eldritch Enlightenment image with the
Kimi-K2.6 DFlash and Eagle3 draft options. This is the successor to
`kimi-k27-code.md`.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629
voipmonitor/vllm@sha256:2ddd49e4ee162ea79a0f2e537aec3e18918bef168e3d684cfa5332151c7fbe2e
```

| Component | Revision |
|---|---|
| vLLM | `codex/eldritch-head66-b12xmla-20260629 @ 8722ac7f8427919ed67bfe9c5e47b3cc30dfbf2e` |
| B12X | `15cd38ce3f10ee5cb7db1179cbc7c88fd15e37b7` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |

See [`eldritch-enlightenment-docker.md`](./eldritch-enlightenment-docker.md) for the full
Docker build recipe and component pins.

## Models

Target:

```text
moonshotai/Kimi-K2.7-Code
```

DFlash draft:

```text
/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
```

Eagle3 draft:

```text
festr2/kimi-k2.6-eagle3-mla-fp8
```

## Runtime

| Setting | Value |
|---|---|
| TP / DCP | `8 / 1` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_ATTN` |
| KV cache | `fp8` |
| Runner | V2 |
| DFlash tokens | `7` |
| Eagle3 tokens | `3` |
| Tool parser | `kimi_k2` |
| Reasoning parser | `kimi_k2` |
| Custom allreduce | disabled for this profile |

Important: with DFlash `num_speculative_tokens=7`, CUDA graph capture sizes
must include a multiple of `8`. `--max-cudagraph-capture-size=4` fails with:

```text
No valid cudagraph sizes after rounding to multiple of 8
```

Use `--max-cudagraph-capture-size=8` or higher. For production testing use
`max_num_seqs=64` and graph cap `64` or higher. `max_num_seqs=1` plus graph cap
`8` is only a fast one-client debug profile.

Eagle3 note: the final image keeps the Eagle draft at its native DCP setting by
default. Do not set `VLLM_DCP_SHARD_DRAFT=1` for Eagle3 unless explicitly
testing that path. Earlier Eldritch images inherited the target DCP value into
the Eagle draft and DCP4 failed during load with a target/draft KV-head
validation error.

## Docker Compose

```yaml
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629}
    container_name: ${NAME:-kimi-k27-code-v2}
    init: true
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k27-code-v2}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TORCH_EXTENSIONS_DIR: /cache/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      TRITON_CACHE_DIR: /cache/triton
      XDG_CACHE_HOME: /cache
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      NCCL_P2P_LEVEL: SYS
      NCCL_IB_DISABLE: "1"
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      SAFETENSORS_FAST_GPU: "1"
      TARGET: ${TARGET:-moonshotai/Kimi-K2.7-Code}
      DRAFT: ${DRAFT:-/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a}
      PORT: ${PORT:-8000}
    command:
      - /bin/sh
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM
        exec vllm serve "$$TARGET" \
          --served-model-name Kimi-K2.7-Code \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --trust-remote-code \
          --tensor-parallel-size 8 \
          --decode-context-parallel-size 1 \
          --kv-cache-dtype fp8 \
          --attention-backend TRITON_MLA \
          --gpu-memory-utilization 0.94 \
          --max-model-len 262144 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 64 \
          --mm-processor-cache-gb 0 \
          --mm-encoder-tp-mode weights \
          --reasoning-parser kimi_k2 \
          --tool-call-parser kimi_k2 \
          --enable-auto-tool-choice \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --speculative-config "{\"model\":\"$$DRAFT\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"
```

## Single Docker Run

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629
DRAFT=/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
CACHE=/root/.cache/vllm-kimi-k27-code-v2

docker rm -f kimi-k27-code-v2 2>/dev/null || true
mkdir -p "$CACHE"

docker run -d --name kimi-k27-code-v2 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$CACHE":/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  "$IMAGE" /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM; exec vllm serve moonshotai/Kimi-K2.7-Code --served-model-name Kimi-K2.7-Code --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --decode-context-parallel-size 1 --kv-cache-dtype fp8 --attention-backend TRITON_MLA --gpu-memory-utilization 0.94 --max-model-len 262144 --max-num-seqs 64 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 64 --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --enable-auto-tool-choice --async-scheduling --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors --speculative-config "{\"model\":\"'$DRAFT'\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"'
```

## Eagle3 Variant

Replace only `--speculative-config`:

```bash
--speculative-config '{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"rejection_sample_method":"standard","draft_sample_method":"greedy"}'
```

For DCP4, keep the target launch identical except
`--decode-context-parallel-size 4`. The final image does not require
`VLLM_DCP_SHARD_DRAFT=0`; this is now the Eagle3 default.

## Validation

Startup and short-context generation are validated on the final image.

A debug run with graph cap `4` fails because DFlash 7 requires graph sizes
rounded to a multiple of 8. Use graph cap `8` or higher.

Measured on 8x RTX 6000 Pro Blackwell, TP8/DCP1, final image. These are
one-client smoke measurements with `max_num_seqs=1` and graph cap `8`; rerun a
full sweep with `max_num_seqs=64` before treating them as production throughput.

Current `8722ac7/b12x8ce61f9` clean-image DFlash7 smoke used target
`TRITON_MLA`, draft `TRITON_ATTN`, V2 runner, KV cache `369,429`, and graph
cap `8`. `/mnt/test.py -L` returned coherent output with `0` CJK and roughly
`200-310 tok/s` generation-only in the debug profile.

| Test | Result |
|---|---:|
| No-spec short smoke | 104.8 tok/s, CJK 0, `finish=stop` |
| No-spec 30k-context smoke | 91.4 tok/s, CJK 0, `finish=stop` |
| DFlash7 short smoke | 219.5 tok/s, CJK 0, `finish=stop` |
| DFlash7 30k-context smoke | 210.6 tok/s, CJK 0, `finish=stop` |
| DFlash7 KV cache budget | 384,320 tokens |
| DFlash7 acceptance, 30k smoke | about `0.89/0.75/0.57/0.49/0.41/0.34/0.28` |

Commands:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /mnt/test.py --port 8000 -c 30000
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```

Additional final-image DCP4 checks, TP8, `max_num_seqs=1`, graph cap `8`:

| Mode | Context | Result |
|---|---:|---:|
| DFlash7 | 0k smoke | 159.3 gen tok/s, CJK 0 |
| DFlash7 | 0k cc1 decode bench | 112.8 tok/s |
| DFlash7 | KV cache budget | 1,604,480 tokens |
| Eagle3 | 0k smoke | 166.8 gen tok/s, CJK 0 |
| Eagle3 | 30k smoke | 169.4 gen tok/s, 58.3 tok/s incl. TTFT, CJK 0 |
| Eagle3 | 0k cc1 decode bench | 115.3 tok/s |
| Eagle3 | KV cache budget | 1,682,944 tokens |

Final image artifacts:

```text
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-none-short.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-none-30k.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-dflash7-short.json
/root/bench-results/final-eldritch-20260626/kimi27-final-dcp1-dflash7-30k.json
/root/bench-results/kimi27-final-bfaa36b-20260627/final-dcp4-dflash7-short-single.json
/root/bench-results/kimi27-final-bfaa36b-20260627/final-dcp4-dflash7-decodebench-cc1.json
/root/bench-results/kimi27-final-bfaa36b-20260627/final-dcp4-eagle3-short-single.json
/root/bench-results/kimi27-final-bfaa36b-20260627/final-dcp4-eagle3-30k-single.json
/root/bench-results/kimi27-final-bfaa36b-20260627/final-dcp4-eagle3-decodebench-cc1.json
```
