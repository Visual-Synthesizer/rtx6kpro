# GLM-5.2 v13 Eldritch NVFP4 / B12X

This page documents the current GLM-5.2 NVFP4 serving stack for RTX 6000 Pro
Blackwell. The default target is Luke's NVFP4 checkpoint, B12X MoE forced to
A16, FP8 KV cache, vLLM V2 model runner, and optional MTP3.

## Image

```text
voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629
voipmonitor/vllm@sha256:2ddd49e4ee162ea79a0f2e537aec3e18918bef168e3d684cfa5332151c7fbe2e
```

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/eldritch-head66-b12xmla-20260629` |
| vLLM commit | `8722ac7f8427919ed67bfe9c5e47b3cc30dfbf2e` |
| B12X branch | `master` |
| B12X commit | `15cd38ce3f10ee5cb7db1179cbc7c88fd15e37b7` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` |
| CUDA / cuBLAS | CUDA `13.2.1`; `libcublas13-cuda-13=13.4.1.2-1`; the base image also carries legacy `libcublas-13-2=13.4.0.1-1` |
| cuDNN / NCCL | cuDNN `9.22.0.52-1`, local NCCL `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| Docker build helper | `/root/vllm/blackwell-llm-docker/build-eldritch-enlightenment-head66-pr20-cu132.sh` |

The image is a clean Docker build, not a runtime overlay.
See [`eldritch-enlightenment-docker.md`](./eldritch-enlightenment-docker.md) for the exact
reproducible build recipe and component pins.

The `8722ac7` build includes the `56fb5d8` Eldritch SM120/DCP stack plus the
GLM/DSA TP6 head66 fix for B12X sparse MLA. The inherited stack includes the
DCP shard-safe warmup prompt fix, native MTP DCP draft sharding, Kimi/MiMo
DFlash fixes, and the upstream GLM sparse-indexer prefill optimization. Earlier
`v0ec1381` images can hang during warmup when the synthetic prompt is shorter
than the DCP shard count.

## Model

```text
lukealonso/GLM-5.2-NVFP4
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

GLM-5.2 still needs the index-cache sparsity pattern override:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

## Backend Choice

For `DCP=1`, the fastest attention path is usually
`FLASHINFER_MLA_SPARSE_SM120`. It should be launched with
`-cc.pass_config.fuse_allreduce_rms=True`; without that pass, the same
DCP1/SM120 recipe can drop from about `81-82 tok/s` to about `75 tok/s` on the
coding smoke test.

As of the `56fb5d8` image, `FLASHINFER_MLA_SPARSE_SM120` can also be used with
DCP. DCP2, DCP4, and DCP8 were validated with MTP disabled, `max_num_seqs=32`,
and coherent short/30k context output. Keep `B12X_MLA_SPARSE` as the default
for MTP3 DCP runs until SM120+MTP is measured separately.

## Runtime Defaults

| Setting | Value |
|---|---|
| TP | `8` |
| DCP | TP8: `1`, `2`, `4`, or `8`; TP6: `1`, `2`, `3`, or `6` |
| MoE backend | `b12x` |
| Quantization | `modelopt_fp4` |
| KV cache | `fp8` |
| A16 | `B12X_MOE_FORCE_A16=1` |
| Decode kernel | `B12X_W4A16_TC_DECODE=1` |
| DCP top-k | enabled by default in this branch |
| DCP draft KV | sharded by default in this branch |
| MTP default | `3`, probabilistic draft sampling |
| Max batched tokens | `8192` |

Do not set `NCCL_GRAPH_FILE` to an empty string. If no XML topology file is
used, unset it before starting vLLM.

`VLLM_DCP_GLOBAL_TOPK` and `VLLM_DCP_SHARD_DRAFT` are legacy debug overrides in
this image; both default to the production-safe enabled mode. They do not need
to be set for normal launches.

## Docker Compose

This compose file supports SM120 or B12X attention. Use
`FLASHINFER_MLA_SPARSE_SM120` for DCP1 or for measured MTP-off DCP runs. Use
`B12X_MLA_SPARSE` for the current production MTP3 DCP recipe.

```yaml
services:
  glm52:
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629}
    container_name: ${NAME:-glm52-v13}
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    init: true
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - ${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-glm52-v13}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      CUTE_DSL_ARCH: sm_120a
      OMP_NUM_THREADS: "16"
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_MOE: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: ${VLLM_ENABLE_PCIE_ALLREDUCE:-1}
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      B12X_DENSE_SPLITK_TURBO: "1"
      B12X_W4A16_TC_DECODE: "1"
      B12X_MOE_FORCE_A16: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TRITON_CACHE_DIR: /cache/jit/triton
      TORCH_EXTENSIONS_DIR: /cache/jit/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/jit/torchinductor
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      XDG_CACHE_HOME: /cache/jit
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}
      PORT: ${PORT:-8000}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-1}
      MTP_TOKENS: ${MTP_TOKENS:-3}
      ATTN_BACKEND: ${ATTN_BACKEND:-FLASHINFER_MLA_SPARSE_SM120}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.955}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-262144}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-32}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}
      GLM52_INDEX_TOPK_PATTERN: FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
    command:
      - /bin/bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        SPEC_ARGS=()
        if [ "$${MTP_TOKENS}" != "0" ]; then
          SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$${MTP_TOKENS},\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}")
        fi
        exec vllm serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --trust-remote-code \
          --tensor-parallel-size "$${TP_SIZE}" \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --quantization modelopt_fp4 \
          --kv-cache-dtype fp8 \
          --attention-backend "$${ATTN_BACKEND}" \
          --moe-backend b12x \
          --load-format fastsafetensors \
          -cc.pass_config.fuse_allreduce_rms=True \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-model-len "$${MAX_MODEL_LEN}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser glm47 \
          --reasoning-parser glm45 \
          --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
          --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"$${GLM52_INDEX_TOPK_PATTERN}\"}" \
          "$${SPEC_ARGS[@]}"
```

## Single Docker Run: DCP1 Fast Path

DCP1 with SM120 attention, B12X MoE, A16 decode, and MTP3:

```bash
docker rm -f glm52-v13 2>/dev/null || true

docker run -d --name glm52-v13 \
  --gpus all --ipc=host --shm-size=32g --network=host --init \
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-glm52-v13:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_W4A16_TC_DECODE=1 \
  -e B12X_MOE_FORCE_A16=1 \
  voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629 \
  /bin/bash -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS; exec vllm serve /root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522 --served-model-name GLM-5.2-NVFP4 --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 8 --decode-context-parallel-size 1 --quantization modelopt_fp4 --kv-cache-dtype fp8 --attention-backend FLASHINFER_MLA_SPARSE_SM120 --moe-backend b12x --load-format fastsafetensors -cc.pass_config.fuse_allreduce_rms=True --gpu-memory-utilization 0.955 --max-model-len 262144 --max-num-seqs 32 --max-num-batched-tokens 8192 --max-cudagraph-capture-size 128 --async-scheduling --enable-chunked-prefill --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --default-chat-template-kwargs "{\"reasoning_effort\":\"high\"}" --hf-overrides "{\"use_index_cache\":true,\"index_topk_pattern\":\"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"moe_backend\":\"b12x\",\"draft_sample_method\":\"probabilistic\"}"'
```

For a no-MTP baseline, remove the trailing `--speculative-config ...` argument.
That baseline is useful because the DCP1/SM120 fast path should be around
`81-82 tok/s` on `/mnt/test.py -L`; with MTP3 and normal acceptance it should
be materially higher.

DCP4 with B12X attention and MTP3: change
`--decode-context-parallel-size 4`, `--attention-backend B12X_MLA_SPARSE`, add
`VLLM_USE_B12X_SPARSE_INDEXER=1`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`,
`VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, and add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic"}'
```

## TP6 Notes

TP6 is supported, but it is not the same runtime profile as TP8:

- Use `B12X_MLA_SPARSE` attention for TP6.
- Use DCP values that divide TP6: `1`, `2`, `3`, or `6`.
- The model-level virtual-TP padding is now minimal for GLM/DSA: TP6 prints
  `attention heads 64 -> 66`. The B12X sparse-MLA backend then pads only the
  kernel-facing tensors from `66 -> 80` and slices outputs back to `66`.
- On this host, B12X PCIe oneshot allreduce with world size 6 did not complete
  startup. The validated TP6 recipe disables it with
  `VLLM_ENABLE_PCIE_ALLREDUCE=0`, so TP6 falls back to PyNCCL allreduce.
- The TP6 validation below used `MAX_MODEL_LEN=128000`,
  `GPU_MEMORY_UTILIZATION=0.957`, `MAX_NUM_BATCHED_TOKENS=2048`,
  `MAX_NUM_SEQS=1`, graph cap `4` without MTP and `16` with MTP3.

Example compose override for TP6/DCP6/MTP3, using the compose file shown
above:

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x15cd38c-cu132-20260629 \
TP_SIZE=6 \
DCP_SIZE=6 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
ATTN_BACKEND=B12X_MLA_SPARSE \
VLLM_ENABLE_PCIE_ALLREDUCE=0 \
GPU_MEMORY_UTILIZATION=0.957 \
MAX_MODEL_LEN=128000 \
MAX_NUM_BATCHED_TOKENS=2048 \
MAX_NUM_SEQS=1 \
MAX_CUDAGRAPH_CAPTURE_SIZE=16 \
MTP_TOKENS=3 \
docker compose -f ./compose.yaml up -d
```

## Validation

The table below is a clean `vfcc6141` coherence smoke, not the full production
throughput sweep. It was measured with `max_num_seqs=1` and graph cap `4` so
that every DCP/MTP combination can be restarted quickly during validation. Full
production launches should use `max_num_seqs=32` and graph cap `128` or larger.

Each row ran the same coding smoke twice: a short prompt and a padded
`-c 30000` long-context prompt. All listed runs returned coherent output with
`0` CJK characters.

| Mode | Attention | MTP | KV cache tokens | Short ctx tok/s | 30k ctx tok/s | 30k TTFT |
|---|---|---:|---:|---:|---:|---:|
| TP8 DCP1 | `FLASHINFER_MLA_SPARSE_SM120` | off | 682,624 | 83.7 | 80.0 | 9.405s |
| TP8 DCP1 | `FLASHINFER_MLA_SPARSE_SM120` | 3 | 652,608 | 163.5 | 167.5 | 9.098s |
| TP8 DCP2 | `B12X_MLA_SPARSE` | off | 1,352,064 | 66.3 | 65.0 | 4.776s |
| TP8 DCP2 | `B12X_MLA_SPARSE` | 3 | 1,289,728 | 97.1 | 86.5 | 4.992s |
| TP8 DCP4 | `B12X_MLA_SPARSE` | off | 2,704,128 | 62.3 | 62.5 | 6.063s |
| TP8 DCP4 | `B12X_MLA_SPARSE` | 3 | 2,579,456 | 91.1 | 83.2 | 6.790s |
| TP8 DCP8 | `B12X_MLA_SPARSE` | off | 5,392,896 | 56.4 | 56.5 | 9.418s |
| TP8 DCP8 | `B12X_MLA_SPARSE` | 3 | 5,143,552 | 89.3 | 85.2 | 10.591s |

SM120 DCP validation used the same Luke NVFP4 checkpoint, B12X MoE A16,
`max_num_seqs=32`, `max_num_batched_tokens=8192`, MTP disabled, and
`gpu_memory_utilization=0.955`. DCP2/DCP4 used graph cap `128`; DCP8 used graph
cap `32` because graph cap `128` did not complete startup during validation.
All rows below produced coherent short and 30k-context Sieve output with `0`
CJK characters.

| Mode | Attention | MTP | Graph cap | KV cache tokens | Short ctx tok/s | 30k ctx tok/s |
|---|---|---:|---:|---:|---:|---:|
| TP8 DCP2 | `FLASHINFER_MLA_SPARSE_SM120` | off | 128 | 1,357,696 | 67.4 | 65.7 |
| TP8 DCP4 | `FLASHINFER_MLA_SPARSE_SM120` | off | 128 | 2,715,392 | 65.7 | 64.5 |
| TP8 DCP8 | `FLASHINFER_MLA_SPARSE_SM120` | off | 32 | 5,415,424 | 62.5 | 61.0 |

TP6 validation used B12X attention and `VLLM_ENABLE_PCIE_ALLREDUCE=0`.

The current `8722ac7/b12x15cd38c` clean image was smoke-tested on
TP6/DCP6/MTP3 with the 78-character `index_topk_pattern`, graph cap `4`, and
`max_num_seqs=1`. Logs showed `attention heads 64 -> 66`, B12X kernel padding
`66 -> 80`, KV cache `1,068,888`, V2 runner, no `topk_scores_buffer` error,
and coherent short plus 30k-context Sieve output with `0` CJK. Generation-only
throughput was about `79-84 tok/s` on short context and `75-79 tok/s` at 30k.

The same image was smoke-tested on TP8/DCP2/MTP3 with `B12X_MLA_SPARSE` and no
explicit `VLLM_DCP_GLOBAL_TOPK` or `VLLM_DCP_SHARD_DRAFT` environment
overrides. It started successfully, reported KV cache `1,289,728`, and
produced coherent `/mnt/test.py -L` output with `0` CJK and about `96-98 tok/s`
generation-only. This validates that the production DCP top-k and sharded-draft
defaults are active without requiring legacy debug envs.

| Mode | Attention | MTP | KV cache tokens | Short ctx tok/s | 30k ctx tok/s | 30k TTFT |
|---|---|---:|---:|---:|---:|---:|
| TP6 DCP1 | `B12X_MLA_SPARSE` | off | 156,672 | 57.8 | 56.1 | 9.109s |
| TP6 DCP1 | `B12X_MLA_SPARSE` | 3 | 130,112 | 115.4 | 92.3 | 4.436s |
| TP6 DCP2 | `B12X_MLA_SPARSE` | off | 320,256 | 46.6 | 46.7 | 6.317s |
| TP6 DCP2 | `B12X_MLA_SPARSE` | 3 | 258,944 | 79.4 | 69.4 | 6.519s |
| TP6 DCP3 | `B12X_MLA_SPARSE` | off | 480,143 | 44.7 | 44.9 | 7.916s |
| TP6 DCP3 | `B12X_MLA_SPARSE` | 3 | 388,221 | 79.4 | 68.5 | 7.952s |
| TP6 DCP6 | `B12X_MLA_SPARSE` | off | 951,952 | 40.1 | 40.5 | 11.721s |
| TP6 DCP6 | `B12X_MLA_SPARSE` | 3 | 768,383 | 78.7 | 75.3 | 12.063s |

Representative MTP3 acceptance logs:

| Mode | Short-context positions | 30k-context positions |
|---|---|---|
| DCP1 MTP3 | about `0.94 / 0.78 / 0.62` | about `0.94 / 0.79 / 0.62` |
| DCP2 MTP3 | about `0.93 / 0.77 / 0.64` | about `0.93 / 0.81 / 0.66` |
| DCP4 MTP3 | about `0.93 / 0.80 / 0.64` | about `0.89-0.94 / 0.79-0.82 / 0.63-0.68` |
| DCP8 MTP3 | about `0.94 / 0.81 / 0.67` | about `0.90-0.94 / 0.75-0.81 / 0.50-0.67` |
| TP6 DCP1 MTP3 | about `0.89 / 0.73 / 0.59` | about `0.95 / 0.84 / 0.72` |
| TP6 DCP2 MTP3 | about `0.95 / 0.83 / 0.65` | about `0.95 / 0.83 / 0.65` |
| TP6 DCP3 MTP3 | about `0.94 / 0.81 / 0.69` | about `0.95 / 0.82 / 0.70` |
| TP6 DCP6 MTP3 | about `0.91 / 0.73 / 0.55` | about `0.93 / 0.80 / 0.68` |

The full decode/prefill benchmark artifacts are stored under:

```text
/root/bench-results/glm52-sm120dcp-v56fb5d8-20260628/
```

Useful checks:

```bash
python3 /mnt/test.py --port 8000 -L
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --concurrency 1 --contexts 0k --duration 30 --skip-prefill
python3 /root/llm-inference-bench/llm_decode_bench.py --port 8000 --prefill-only --standalone-prefill --prefill-contexts 8k,64k --prefill-duration 10
```
