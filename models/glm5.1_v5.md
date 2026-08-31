# GLM-5.1 v5 on 8x RTX PRO 6000 Blackwell

Measured on 2026-05-26 on the local 8-GPU RTX PRO 6000 Blackwell host.

This page documents the GLM-5.1 v5 rebuild based on upstream vLLM main plus the
local GLM/B12X stack. The image uses the MTP checkpoint as both the target model
and the speculative draft model. Do not substitute the non-MTP checkpoint for
validation runs.

Current status: DCP1, DCP2, DCP4, and DCP8 start and benchmark correctly with
MTP and W4A16 decode. Do not override `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` for
DCP profiles; leave it unset so B12X auto-calculates the DCP scratch budget.
Forcing it to `32` recreates the startup illegal-address failure described
below.

## Docker Compose

Save this as `compose.glm51-v5.yml`:

```yaml
services:
  glm51-v5:
    image: voipmonitor/vllm:glm51-v5-upstreammain-vllm4cdbe04-b12xf6abdd2-flashinfer56d537a-20260526
    container_name: glm51-v5
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: /bin/bash
    command: -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE; exec /opt/vllm/scripts/run-glm51-vllm'
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      OMP_NUM_THREADS: "16"
      SAFETENSORS_FAST_GPU: "1"
      CUTE_DSL_ARCH: sm_120a
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_DISABLE: "0"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_B12X_MLA_SPEC_SERIAL_DECODE: "0"
      VLLM_MTP_RETURN_NORMALIZED_HIDDEN: "1"
      VLLM_SPEC_ACCEPT_THRESHOLD_ACC: "1.0"
      VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE: "1.0"
      VLLM_B12X_FORCE_MOE_A16: "1"
      B12X_MOE_FORCE_A16: "0"
      PORT: "${PORT:-5317}"
      SERVED_MODEL_NAME: GLM-5
      MODEL: /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
      SPEC_CONFIG: '{"model":"/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989","method":"mtp","num_speculative_tokens":3,"moe_backend":"b12x","draft_sample_method":"probabilistic","use_local_argmax_reduction":false}'
      TP_SIZE: "8"
      DCP_SIZE: "${DCP_SIZE:-1}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.855}"
      MAX_MODEL_LEN: "202752"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "64"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "256"
      KV_CACHE_DTYPE: fp8
      ATTENTION_BACKEND: B12X_MLA_SPARSE
      MOE_BACKEND: b12x
      GLM51_DISABLE_MTP: "0"
      GLM51_USE_LOCAL_ARGMAX_REDUCTION: "0"
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
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${HOME}/.cache/vllm-glm51-v5-serve/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${HOME}/.cache/vllm-glm51-v5-serve/jit:/cache/jit
      - ${HOME}/.cache/vllm-glm51-v5-serve/triton:/root/.cache/triton
      - ${HOME}/.cache/vllm-glm51-v5-serve/torchinductor:/root/.cache/torchinductor
      - ${HOME}/.cache/vllm-glm51-v5-serve/vllm:/root/.cache/vllm
```

Start DCP1:

```bash
GPU_MEMORY_UTILIZATION=0.855 docker compose -f compose.glm51-v5.yml up -d
curl -fsS http://127.0.0.1:5317/health
```

Start DCP2:

```bash
DCP_SIZE=2 GPU_MEMORY_UTILIZATION=0.855 docker compose -f compose.glm51-v5.yml up -d
curl -fsS http://127.0.0.1:5317/health
```

Start DCP4:

```bash
DCP_SIZE=4 GPU_MEMORY_UTILIZATION=0.845 docker compose -f compose.glm51-v5.yml up -d
curl -fsS http://127.0.0.1:5317/health
```

Start DCP8:

```bash
DCP_SIZE=8 GPU_MEMORY_UTILIZATION=0.835 docker compose -f compose.glm51-v5.yml up -d
curl -fsS http://127.0.0.1:5317/health
```

Do not set or override `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` for DCP profiles.
Leaving it unset lets the B12X MLA backend auto-calculate the DCP-safe value
from the scratch budget.

Do not set `NCCL_GRAPH_FILE=` as an empty environment variable. For this image,
leave it unset. The compose command explicitly unsets both `NCCL_GRAPH_FILE` and
`NCCL_GRAPH_DUMP_FILE` before starting vLLM.

## Build

The measured image was built from:

```bash
git clone --branch codex/vllm-upstream-main-stack-20260526 https://github.com/voipmonitor/vllm.git vllm-glm51-v5
cd vllm-glm51-v5
git rev-parse HEAD
# expected: 4cdbe047c596342f2e924101798cd907469d0090

docker build \
  -f docker/Dockerfile.glm51-kimi-b12x013 \
  --build-arg B12X_REPO=https://github.com/voipmonitor/b12x.git \
  --build-arg B12X_COMMIT=f6abdd287994141712f8401645afcc3e4b25dbc8 \
  --build-arg FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git \
  --build-arg FLASHINFER_COMMIT=56d537a106024eb25f4d4a186eadc226990a9185 \
  --build-arg VLLM_BUILD_VERSION=0.11.2.dev278+glm51v520260526 \
  -t voipmonitor/vllm:glm51-v5-upstreammain-vllm4cdbe04-b12xf6abdd2-flashinfer56d537a-20260526 \
  .
```

## Source State

| Component | Value |
|---|---|
| Docker tag | `voipmonitor/vllm:glm51-v5-upstreammain-vllm4cdbe04-b12xf6abdd2-flashinfer56d537a-20260526` |
| Image ID | `sha256:8e0db3ca66cf27191c413a176f675e6072b04e99a3cf6e58c91a096f35106cb5` |
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | [`codex/vllm-upstream-main-stack-20260526`](https://github.com/voipmonitor/vllm/tree/codex/vllm-upstream-main-stack-20260526) |
| vLLM commit | [`4cdbe047c596342f2e924101798cd907469d0090`](https://github.com/voipmonitor/vllm/commit/4cdbe047c596342f2e924101798cd907469d0090) |
| B12X repo | `https://github.com/voipmonitor/b12x.git` |
| B12X branch | `codex/explicit-w13-order-cleanup-20260525` |
| B12X commit | `f6abdd287994141712f8401645afcc3e4b25dbc8` |
| FlashInfer repo | `https://github.com/flashinfer-ai/flashinfer.git` |
| FlashInfer commit | `56d537a106024eb25f4d4a186eadc226990a9185` |
| vLLM runtime version | `0.11.2.dev278+glm51v520260526` |
| FlashInfer runtime version | `0.6.12+cu130` |
| Transformers in image | `5.3.0` |

The Docker labels include an older `flashinfer.version` label, but runtime import
inside this image reports `flashinfer-python==0.6.12+cu130` from the git build
above. Use the runtime import and `voipmonitor.flashinfer.git_sha` label as the
source of truth.

## Transformers 5.9.0 Smoke Test

A disposable container was used to install `transformers==5.9.0` and load the
MTP snapshot config/tokenizer:

```text
transformers 5.9.0
config GlmMoeDsaConfig glm_moe_dsa ['GlmMoeDsaForCausalLM']
tokenizer TokenizersBackend 154820
```

This validates basic Hugging Face config/tokenizer loading with the latest pip
release, not full vLLM runtime compatibility. The v5 image itself still carries
`transformers==5.3.0`.

## KLD Validation

Correct target checkpoint:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

Invalid validation artefacts using the non-MTP checkpoint were discarded and
must not be used for v5 conclusions.

Prefill KLD against BF16:

| Test | Value |
|---|---:|
| Mean KLD | `0.098751` |
| Positions | `2047` |
| Positions/sec | `11.90` |
| Log | `/root/kld/v5_nvfp4_mtp_a16_prefill_kld_vs_bf16_vllm_ctx2048_w1_20260526_112216.log` |

Important interpretation: with `VLLM_B12X_FORCE_MOE_A16=1`, this stack uses
B12X W4A16 MoE only for decode-only forwards. Prefill and mixed batches keep
NVFP4 activation numerics, so this prefill number does not validate the A16
decode kernel path.

Teacher-forced decode KLD against BF16:

| Test | Value |
|---|---:|
| BF16 -> variant KL | `0.000015627370885340497` |
| variant -> BF16 KL | `0.000020867075363639742` |
| JS | `0.000004202187483315356` |
| Positions | `16` |
| Tensor | `/root/kld/decode_teacher_v5_nvfp4mtp_a16_ctx2048_t17_20260526_112606.safetensors` |
| Compare JSON | `/root/kld/decode_teacher_v5_nvfp4mtp_a16_ctx2048_t17_20260526_112606_vs_bf16_teacher_compare.json` |

For teacher-forced decode, the primary signal is BF16 -> variant KL. The reverse
KL is kept in the artefact for diagnostics, but it is not the primary quality
metric for this test because the student is snapped back to the BF16/reference
token at every step.

## Decode Benchmark

Benchmark command template:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5317 \
  --model GLM-5 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --dcp-size "${DCP_SIZE}" \
  --display-mode plain \
  --output "/root/bench-results/glm51-v5-upstreammain-20260526/dcp${DCP_SIZE}-mtp1-a16-full/full.json"
```

`llm_decode_bench.py` version:

```text
0.4.23
```

All full-sweep runs below used MTP enabled, `VLLM_B12X_FORCE_MOE_A16=1`,
`VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, `VLLM_USE_B12X_SPARSE_INDEXER=1`,
`CUTE_DSL_ARCH=sm_120a`, and `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` unset.

DCP1 quick validation was able to run at `GPU_MEMORY_UTILIZATION=0.865`, but a
full CUDA graph capture run at that setting failed with a normal OOM while
allocating 448 MiB. The full sweep therefore uses `0.855`.

Full-sweep artefacts:

| DCP | GPU mem util | File |
|---:|---:|---|
| 1 | `0.855` | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp1-mtp1-a16-full-gpu0855/full-sweep-20260526-1818.json` |
| 2 | `0.855` | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp2-mtp1-a16-full-gpu0855/full-sweep-20260526-1836.json` |
| 4 | `0.845` | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp4-mtp1-a16-noextendoverride/full-sweep-20260526-1506.json` |
| 8 | `0.835` | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp8-mtp1-a16-full-gpu0835/full-sweep-20260526-1851.json` |

Quick A4 vs A16 decode comparison:

These are quick `ctx=0` checks for `cc=1` and `cc=16`, not full sweeps. All
runs used MTP enabled and the same v5 image/config family. `A4` means
`VLLM_B12X_FORCE_MOE_A16=0` and `B12X_MOE_FORCE_A16=0`; `A16` means
`VLLM_B12X_FORCE_MOE_A16=1`.

| DCP | MoE decode mode | GPU mem util | cc1 tok/s | cc16 tok/s | File |
|---:|---|---:|---:|---:|---|
| 1 | A4 | `0.855` | 87.8 | 561.3 | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp1-mtp1-a16off-quick-gpu0855/quick-cc1-cc16-ctx0-20260526-1932.json` |
| 1 | A16 | `0.855` | 85.8 | 558.1 | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp1-mtp1-a16/quick-cc1-cc16-ctx0.json` |
| 4 | A4 | `0.845` | 69.6 | 383.8 | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp4-mtp1-a16off-quick-gpu0845/quick-cc1-cc16-ctx0-20260526-1938.json` |
| 4 | A16 | `0.845` | 65.0 | 389.5 | `/root/bench-results/glm51-v5-upstreammain-20260526/dcp4-mtp1-a16-noextendoverride/quick-cc1-cc16-ctx0.json` |

MTP-off DCP4 A4 regression check, 2026-05-27:

The apparent DCP4 MTP-off gap against v2 was mostly an apples-to-oranges
comparison. The stored v2 `glm-dcp4-mtp0` artefact was not pure A4; its
`docker-run.env` contains `B12X_MOE_FORCE_A16=1`. Re-running the v2 image with
A16 explicitly disabled gives the correct A4 reference.

| Stack | Mode | cc1 tok/s | cc16 tok/s | File |
|---|---|---:|---:|---|
| v2 stored artefact | MTP off, legacy A16 env | 52.15 | 336.54 | `/root/bench-results/glm51-v2-full-68b3569f-20260514/glm-dcp4-mtp0/decode-matrix.json` |
| v2 re-run | MTP off, A4 | 51.0 | 322.7 | `/root/bench-results/glm51-v5-mtp0-a4-regression-20260527/v2-reference-dcp4-mtp0-a4-cppar56-gpu0865/quick-cc1-cc16-ctx0-duration30.json` |
| v5 baseline | MTP off, A4 | 48.0 | 309.4 | `/root/bench-results/glm51-v5-mtp0-a4-regression-20260527/main-dcp4-mtp0-a4-cppar56-noautotune-gpu0865/quick-cc1-cc16-ctx0.json` |
| v5 + B12X fused final-LSE | MTP off, A4 | 50.8 | 320.1 | `/root/bench-results/glm51-v5-mtp0-a4-regression-20260527/main-dcp4-mtp0-a4-fusedlse-cppar56-noautotune-gpu0865/quick-cc1-cc16-ctx0-duration30-numchunkptr.json` |
| v5 + B12X fused final-LSE | MTP off, A16 | 43.8 | 346.0 | `/root/bench-results/glm51-v5-mtp0-a4-regression-20260527/main-dcp4-mtp0-a16-fusedlse-cppar56-noautotune-gpu0865/quick-cc1-cc16-ctx0-duration30.json` |

The real v5 DCP4 A4 loss was the B12X sparse MLA final-LSE path. v2 used a
small vLLM Triton final-LSE kernel; v5 delegated `return_lse=True` into B12X,
where final LSE was computed by a generic PyTorch `logsumexp` over split chunks
for every decode layer. B12X commit
[`9cf67f7`](https://github.com/voipmonitor/b12x/commit/9cf67f7) adds a fused
Triton final-LSE reduction using the existing B12X split-workspace tensors and
recovers v5 DCP4 MTP-off A4 to the v2 A4 level. AR backend and FlashInfer
autotune were checked separately and were not the root cause.

The experimental variant that moved final-LSE before split-MLA merge was
rejected: it measured `50.8` / `316.1` tok/s and was not committed.

Prefill scout speed:

| DCP | 8k tok/s | 16k tok/s | 32k tok/s | 64k tok/s | 128k tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 3,624 | 4,304 | 4,116 | ∅ | ∅ |
| 2 | 2,892 | 3,371 | 3,278 | 3,155 | ∅ |
| 4 | 2,687 | 2,496 | 2,383 | 2,306 | 2,234 |
| 8 | 1,750 | 1,717 | 1,529 | 1,444 | 1,398 |

Aggregate sustained decode tok/s, DCP1:

| ctx / concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 84.8 | 141.5 | 233.2 | 360.0 | 552.3 | ∅ | ∅ | ∅ |
| 16k | 81.2 | 136.5 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 32k | 82.3 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

Aggregate sustained decode tok/s, DCP2:

| ctx / concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 68.1 | 117.3 | 189.9 | 288.8 | 451.7 | 596.1 | ∅ | ∅ |
| 16k | 64.5 | 112.0 | 175.6 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 32k | 65.3 | 112.8 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | 65.1 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

Aggregate sustained decode tok/s, DCP4:

| ctx / concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 68.1 | 106.6 | 174.2 | 251.9 | 385.1 | 497.8 | 636.7 | ∅ |
| 16k | 64.6 | 101.9 | 161.9 | 240.9 | ∅ | ∅ | ∅ | ∅ |
| 32k | 62.1 | 100.2 | 157.5 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 64k | 61.3 | 103.7 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 61.3 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

Aggregate sustained decode tok/s, DCP8:

| ctx / concurrency | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 61.7 | 96.5 | 150.1 | 213.6 | 293.4 | 345.9 | 456.5 | 437.2* |
| 16k | 57.5 | 90.2 | 141.4 | 197.3 | 258.8 | ∅ | ∅ | ∅ |
| 32k | 59.3 | 92.2 | 135.8 | 195.8 | ∅ | ∅ | ∅ | ∅ |
| 64k | 60.0 | 89.8 | 135.0 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 58.1 | 88.7 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

`∅` means the cell was skipped or hidden because it does not fit in the KV
cache for this run. `*` means the JSON marks the cell as capacity-limited or
warmup-limited, so it is not a fully admitted steady-state comparison point.
Exact capacity details are in the JSON artefacts.

## DCP4 Extend-Chunk Trap

The original DCP4 failure was caused by adding this override:

```yaml
VLLM_B12X_MLA_EXTEND_MAX_CHUNKS: "32"
```

With DCP4, the B12X MLA split path allocates tmp output after Q-head all-gather.
For `max_num_batched_tokens=8192`, 32 gathered Q heads, `v_head_dim=512`, and
BF16 tmp output, one chunk is 256 MiB. Forcing 32 chunks creates an 8 GiB split
tmp-output layout per worker. A B12X-only reproducer failed in
`run_sparse_mla_split_decode_forward` with the fixed scratch shape
`[8192, 32, 32, 512]`.

Leaving `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` unset uses the built-in DCP budget:
2 GiB / 256 MiB = 8 chunks. That starts cleanly with a 2.21 GiB B12X joint
arena and keeps the API healthy.

Successful DCP4 startup at `GPU_MEMORY_UTILIZATION=0.845` without the override:

```text
B12X joint arena allocation: shared=2.21 GiB
GPU KV cache size: 878,335 tokens
Maximum concurrency for 202,752 tokens per request: 4.33x
```

The failing run with `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS=32` reached:

```text
Model loading took 70.14 GiB memory
Available KV cache memory: 7.58 GiB
GPU KV cache size: 523,008 tokens
Maximum concurrency for 202,752 tokens per request: 2.58x
```

It then fails during runtime prewarm before the API server becomes healthy:

```text
Runtime Triton prewarm: slot-mapping kernel warmup failed: CUDA error: an illegal memory access was encountered
Runtime Triton prewarm: padded spec-decode kernel warmup failed: CUDA error: an illegal memory access was encountered
Runtime Triton prewarm: rejection sampler kernel warmup failed: CUDA error: an illegal memory access was encountered
Runtime Triton prewarm: B12X paged-MQA schedule warmup failed: CUDA error: an illegal memory access was encountered
RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered'
```

DCP2 at `GPU_MEMORY_UTILIZATION=0.865` failed earlier with a normal OOM during
CUDA graph capture. Lowering DCP2 to `0.855` fixed that. The DCP4
extend-chunk failure was different: it was a B12X split-MLA scratch layout
problem triggered by the forced 32-chunk override, not a KV-cache headroom
problem.
