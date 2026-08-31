# GLM-5.1 v9 BF16 TP16 DCP8 on 16x RTX PRO 6000 Blackwell

Measured on 2026-06-07 on the local 16-GPU RTX PRO 6000 Blackwell host.

Status: BF16 GLM-5.1 starts and serves on all 16 GPUs with TP16, DCP8,
`B12X_MLA_SPARSE`, V2 model runner, CUDA graphs enabled, no explicit
`--max-model-len`, and KV cache reduced to leave request-time FlashInfer MoE
workspace headroom.

## Image

Use this Docker image:

```text
voipmonitor/vllm:abyssal-abjuration-611a842-a16-dcp
```

Pinned digest:

```text
voipmonitor/vllm@sha256:8601786e427faa72368e3d57e04d30a80a33bfbf5372352bdfb4358667827f36
```

Local image ID:

```text
sha256:6befe4f8812337fd0474cd81280d7e4d783c07c7b3bb05c223c5ba94e8254c98
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS package in image env | `13.4.0.1-1` |
| cuDNN package in image env | `9.20.0.48-1` |
| NCCL runtime preload | `/opt/libnccl-local-inference.so.2.30.4` |
| PyTorch | `2.12.0+cu132` |
| vLLM image branch | `dev/abyssal-abjuration` |
| vLLM image commit | `611a842dc1052772b22e5ac48b2da28ced6dfba9` |
| B12X image commit | `f9226c99384b8f7a169e6f5d6251f783886ef775` |
| DCP sparse indexer fix | included in the Docker tag |

## Model

BF16 checkpoint:

```text
/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
```

Served model name:

```text
GLM-5.1-BF16
```

## Runtime Profile

Use this exact profile for the working 16-GPU BF16 DCP8 service:

| Setting | Value |
|---|---|
| GPUs | `0-15` |
| Tensor parallel | `16` |
| Decode context parallel | `8` |
| MTP | off |
| Load format | `safetensors` |
| Quantization | none, BF16 weights |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `auto` |
| KV cache dtype | `fp8` |
| GPU memory utilization | `0.976` |
| Max model len | not explicitly set |
| Runtime max seq len | `202,752` |
| Max num seqs | `1` |
| Max batched tokens | `1,024` |
| Max CUDA graph capture size | `1` |
| CUDA graph mode | `FULL_AND_PIECEWISE` |
| AOT envs | disabled for this run |

The `0.976` KV setting is intentional. Higher values started but later failed
inside FlashInfer fused-MoE workspace allocation. At `0.984`, one failure tried
to allocate `194 MiB` with only about `178 MiB` free. The working run leaves
about `1.15 GiB` free per GPU after startup.

## Required Runtime Overlay

The image contains the DCP sparse indexer fix, but the stable BF16 TP16/DCP8 run
also used a temporary `shared_experts.py` overlay:

```text
/root/vllm/worktrees/vllm-abyssal-abjuration-20260605/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
```

Mount it read-only over:

```text
/opt/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
```

The overlay adds `VLLM_SHARED_EXPERTS_DROP_STALE_OUTPUT=1`, which clears stale
shared-expert output before recomputation. Without this guard, the same profile
previously failed at:

```text
torch.ops.vllm.moe_forward_shared -> SharedExperts.apply
AssertionError: assert self._output[self._output_idx] is None
```

## Launch Script

This is the exact launcher shape used for the working instance:

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:abyssal-abjuration-611a842-a16-dcp}"
NAME="${NAME:-glm51-bf16-tp16-dcp8-nomtp-abyssal}"
MODEL="${MODEL:-/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d}"
PORT="${PORT:-5329}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /root/kld:/root/kld \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v /root/bench-results:/root/bench-results \
  -v /root/vllm/worktrees/vllm-abyssal-abjuration-20260605/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:/opt/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:ro \
  -e MODEL="${MODEL}" \
  -e PORT="${PORT}" \
  -e DCP_SIZE=8 \
  -e MAX_MODEL_LEN= \
  -e LOAD_FORMAT=safetensors \
  -e ENFORCE_EAGER=0 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 \
  -e VLLM_USE_B12X_MOE=1 \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
  -e VLLM_SHARED_EXPERTS_DROP_STALE_OUTPUT=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_W4A16_TC_DECODE=0 \
  -e B12X_MOE_FORCE_A16=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  --entrypoint bash \
  "${IMAGE}" -lc '
set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
export PYTHONPATH="/opt/vllm${PYTHONPATH:+:${PYTHONPATH}}"
cd /opt/vllm
exec /opt/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name GLM-5.1-BF16 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend ag_rs \
  --dcp-kv-cache-interleave-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format "${LOAD_FORMAT}" \
  --async-scheduling \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization 0.976 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --max-cudagraph-capture-size 1 \
  --attention-backend B12X_MLA_SPARSE \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --hf-overrides "{\"index_topk_pattern\":\"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS\"}"
'
```

Important: the image environment contains `NCCL_GRAPH_FILE=`. The launch command
must unset it before `vllm serve`; an empty value can make NCCL try to open an
empty XML graph path.

## Expected Startup Lines

A healthy startup should include:

```text
Initializing a V1 LLM engine ... dtype=torch.bfloat16, max_seq_len=202752,
tensor_parallel_size=16, decode_context_parallel_size=8, enforce_eager=False
Using V2 Model Runner
Using AttentionBackendEnum.B12X_MLA_SPARSE backend.
Available KV cache memory: 1.49 GiB
GPU KV cache size: 208,383 tokens
Maximum concurrency for 202,752 tokens per request: 1.03x
Graph capturing finished ... took 0.11 GiB
Application startup complete.
```

After startup, `nvidia-smi` showed approximately:

```text
96.1 GiB used / 1.15 GiB free per GPU
```

## Smoke Tests

Use these commands:

```bash
curl -fsS http://127.0.0.1:5329/health

python3 /mnt/test.py \
  --port 5329 \
  --model GLM-5.1-BF16 \
  --max-tokens 80 \
  --quiet \
  --json-summary -

python3 /mnt/test.py \
  --port 5329 \
  --model GLM-5.1-BF16 \
  --max-tokens 512 \
  --quiet \
  --json-summary -
```

Validated results:

| Test | Result |
|---|---|
| Health | HTTP OK |
| 80-token smoke | HTTP 200, CJK count `0`, TTFT about `0.21s` |
| 6x sequential 80-token smoke | all HTTP 200, CJK count `0`, TTFT about `0.166-0.170s` |
| 512-token smoke | HTTP 200, CJK count `0`, about `45.9 tok/s` |

Useful log check:

```bash
docker logs glm51-bf16-tp16-dcp8-nomtp-abyssal 2>&1 \
  | tr '\r' '\n' \
  | rg 'GPU KV cache|Maximum concurrency|Application startup|POST /v1/chat|500 Internal|AssertionError|CUDA out of memory|RuntimeError|MemoryError' \
  | tail -120
```

The validated run had no `CUDA out of memory`, `AssertionError`,
`RuntimeError`, or HTTP 500 after the smoke tests.

## Known Bad Settings

Do not repeat these unless the underlying MoE/shared-experts behavior changes:

| Setting | Result |
|---|---|
| `GPU_MEMORY_UTILIZATION=0.987` | KV passed at `354,304` tokens but startup later OOMed in FlashInfer fused-MoE workspace/autotuner. |
| `GPU_MEMORY_UTILIZATION=0.984` without stale-output guard | Startup passed but later failed at `SharedExperts.apply` stale-output assertion. |
| `GPU_MEMORY_UTILIZATION=0.984` with stale-output guard | Short smokes passed, then request-time FlashInfer fused-MoE workspace OOMed on a `194 MiB` allocation. |
| `fastsafetensors` for BF16 | OOMed during BF16 weight load; use `safetensors`. |

