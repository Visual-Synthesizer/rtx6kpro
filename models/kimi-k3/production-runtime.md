# Kimi-K3 MXFP4 Runtime

Status: **qualified** on 16 RTX PRO 6000 Blackwell GPUs.

This runtime serves the official Kimi-K3 MXFP4 checkpoint with three supported
decode profiles:

- Inferact DSpark with five-image input, prefix caching, and native vLLM host
  KV offload;
- target-only decode without speculation;
- modal-labs DFlash speculative decode.

The DSpark profile is the production default. It exposes a 1,000,000-token
request limit and 1,016,293 physical FP8 KV tokens while retaining 32 GiB of
host RAM for native vLLM KV offload.

The machine-readable qualification record is
[`validation/kimi-k3-upstream-aligned-r29-20260821.json`](validation/kimi-k3-upstream-aligned-r29-20260821.json).

## Immutable artifacts

| Component | Identity |
|---|---|
| vLLM image | `voipmonitor/vllm@sha256:f444688fafa4e4649481cec660a3941d20984a2792ca0da3e7b8df42a04135c9` |
| Image tag | `voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllm036b9ad-b12xf006681-cu133-torch213-20260821-r29` |
| Image ID | `sha256:6c3c1514de7061562fe535679448138f59dbc19d67b6e582e678155b35c4322b` |
| Docker recipe and `main` publication | `local-inference-lab/blackwell-llm-docker@bb3a4f954cfcc831dd4520a883a402eb09e66e62` |
| vLLM integration tree | `036b9adf727ff9993335ef688531cdacdbe628a0` |
| B12X integration tree | `f0066813bd55e6e19b6a8d84ab47087510c12890` |
| LLMConduit image | `voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61` |

The vLLM image uses CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL
4.6.2, FlashInfer 0.6.18, InstantTensor 0.1.9, and xgrammar 0.2.5. It contains
installed packages and does not mount or activate a source overlay. The
sixteen-rank NCCL collective smoke test passes.

## Model representation

| Property | Value |
|---|---|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Routed expert weights | checkpoint MXFP4 |
| Selected target dense projections | online MXFP8 weight-only representation |
| DSpark checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| DFlash checkpoint | `modal-labs/Kimi-K3-DFlash@c192d15a43407bf758b5ae0880d5c72052fef1de` |
| Tensor parallelism | 16 |
| Decode-context parallelism | 16 |
| Target KV dtype | FP8 |
| Scheduler chunk | 4,096 tokens |
| Active sequence limit | 1 |

The target dense overlay quantizes KDA Q, K, V, B, and F-A projections. The
vision tower and multimodal projector use the same runtime MXFP8 dense-linear
path. Routed experts remain in the official checkpoint's MXFP4 format.

## Start the production DSpark service

The Hugging Face cache must contain the pinned target and Inferact DSpark
snapshots. Do not set `NCCL_GRAPH_FILE` to an empty value.

```bash
IMAGE=voipmonitor/vllm@sha256:f444688fafa4e4649481cec660a3941d20984a2792ca0da3e7b8df42a04135c9
CACHE_DIR=/mnt/luke/kimi-k3-cache/kimi-k3-r29

mkdir -p "$CACHE_DIR"
docker pull "$IMAGE"

docker run -d \
  --name kimi-k3-r29-production-dspark \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt label=disable \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE_DIR":/cache/jit:rw \
  -e PORT=8001 \
  -e TP_SIZE=16 \
  -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1000000 \
  -e MAX_NUM_BATCHED_TOKENS=4102 \
  -e MAX_NUM_SEQS=1 \
  -e KV_CACHE_MEMORY_BYTES=1325000000 \
  -e MAMBA_BLOCK_SIZE=12288 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HOST_KV_BACKEND=native \
  -e KV_OFFLOADING_SIZE=32 \
  -e ENABLE_PREFIX_CACHING=1 \
  -e ENABLE_VISION=1 \
  -e MAX_IMAGES_PER_PROMPT=5 \
  "$IMAGE" \
  --max-num-scheduled-tokens 4096
```

Readiness and model identity:

```bash
docker logs -f kimi-k3-r29-production-dspark
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

The served model identifier is `Kimi-K3-MXFP4-DSpark7-DCP16-1M`. The TP16
InstantTensor loader streamed 1.41 TiB of target checkpoint data in 162 seconds
at a reported 9.57 GB/s. Complete target and draft model loading finished in
192.72 seconds.

## Start LLMConduit

Copy [`configs/llmconduit-production.yaml`](configs/llmconduit-production.yaml)
to a stable host path and enable turn capture:

```bash
mkdir -p /mnt/luke/kimi-k3-runs/llmconduit-turn-captures

docker run -d \
  --name llmconduit-kimi-k3-production \
  --restart unless-stopped \
  --network host \
  -v /root/vllm/kimi/llmconduit-kimi-k3-production.yaml:/config/config.yaml:ro \
  -v /mnt/luke/kimi-k3-runs/llmconduit-turn-captures:/captures/turns:rw \
  voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61 \
  start --config /config/config.yaml

curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8003/v1/models | jq .
```

Port 8001 is the vLLM API and port 8003 is the LLMConduit API. LLMConduit
accepts OpenAI Chat Completions and Responses requests, streams tool calls,
preserves Kimi reasoning internally, and supports native image inputs. The
profile permits completion output up to 128,000 tokens.

Reasoning-level mapping:

| Client value | Kimi value | Returned reasoning |
|---|---|---|
| `none` | `none` | suppressed |
| `minimal`, `low` | `low` | preserved |
| `medium`, `high` | `high` | preserved |
| `xhigh`, `max` | `max` | preserved |
| omitted | `high` | preserved |

## Configure Oh My Pi

Status: **qualified** with OMP 17.3.5.

```bash
mkdir -p /root/.omp/agent
cp models/kimi-k3/configs/omp-models.yml /root/.omp/agent/models.yml
cp models/kimi-k3/configs/omp-config.yml /root/.omp/agent/config.yml

omp models --json
omp --thinking high
```

The configuration files must be global under `/root/.omp/agent` for OMP to use
the same provider outside a project directory. Supported CLI values are
`minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. OMP retains at most five
images for this provider, matching the vLLM request limit.

## Target-only profile

The target-only launcher is text-only and does not enable host KV offload. It
provides 1,458,823 physical FP8 KV tokens with the qualified manual allocation.

```bash
docker run -d \
  --name kimi-k3-r29-nospec \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE_DIR":/cache/jit:rw \
  -e PORT=8001 -e TP_SIZE=16 -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1000000 -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e MAX_NUM_SEQS=1 -e KV_CACHE_MEMORY_BYTES=1325000000 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii \
  "$IMAGE" --max-num-scheduled-tokens 4096
```

## DFlash profile

The DFlash launcher is text-only. It uses the pre-normalization AttnRes stream
expected by the modal-labs drafter and provides exactly 1,048,576 physical FP8
KV tokens.

```bash
docker run -d \
  --name kimi-k3-r29-dflash \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE_DIR":/cache/jit:rw \
  -e PORT=8001 -e TP_SIZE=16 -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1048576 -e MAX_NUM_BATCHED_TOKENS=2048 \
  -e MAX_NUM_SEQS=1 -e KV_CACHE_MEMORY_BYTES=1200000000 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii \
  "$IMAGE"
```

## Qualification results

The normalized decode measurements use a pinned 256-token prompt, 1,024 output
tokens, temperature zero, and measured requests after warmup. Target cycles per
second isolate target execution speed from prompt-dependent speculative
acceptance.

| Profile | Runs | Emitted median | Acceptance median | Target cycles/s median | Physical KV tokens |
|---|---:|---:|---:|---:|---:|
| Target-only | 7 | 55.719 tok/s | not applicable | not applicable | 1,458,823 |
| Inferact DSpark | 7 | 118.076 tok/s | 0.396534 | 31.416 | 1,016,293 |
| modal-labs DFlash | 5 | 166.715 tok/s | 0.673583 | 29.160 | 1,048,576 |

The immutable r26 control measured 55.693 tok/s for target-only decode and
31.417 target cycles/s for DSpark. The r27 differences are +0.05% and -0.005%,
respectively. A clean, deterministic DFlash A/B measured 29.195 target cycles/s
on r26 and 29.160 on r27, a -0.12% difference. The r29 source delta executes
only for vision requests, so the text-only target, DSpark, DFlash, and prefill
performance qualification remains applicable.

Decode qualification must not run concurrent NVML polling. On the unchanged
r26 DFlash process, starting `nvidia-smi dmon` reduced target execution from
29.195 to 28.136 cycles/s; subsequent samples remained at 28.156 cycles/s.
Measurements affected by that GPU-state transition are excluded from the
qualification record.

Uncached DSpark prefill uses a unique cache salt and disables host offload for
each request:

| Prompt tokens | Median effective prefill |
|---:|---:|
| 8,192 | 3,627.05 tok/s |
| 32,768 | 3,694.53 tok/s |
| 65,535 | 3,538.45 tok/s |

A DFlash request containing 500,224 stored token IDs completed prefill and
eight decode positions in 261.33 seconds. The response returned HTTP 200, all
44 captured log-probability values were finite, and no Kimi protocol marker was
emitted. The r26 qualification record retains the independent 134,209-token,
five-image native host-KV replay evidence.

## Recurrent prefix-hit correctness

Model Runner V2 stores recurrent state on the logical Mamba checkpoint grid.
For this profile, physical attention pages contain 768 tokens and recurrent
checkpoints contain 12,288 tokens. A resumed request must therefore select its
recurrent-state block-table column with the 12,288-token cadence.

[Local vLLM #463](https://github.com/local-inference-lab/vllm/pull/463)
enforces that invariant. Without the change, a request resumed at 110,592
computed tokens selected recurrent column 143 instead of column 8. The
following prefill step could copy unrelated state and emit malformed Kimi
protocol tokens or incoherent text.

The production DSpark profile was qualified with two cache-hit tests:

- An identical 112,301-token OMP request containing five images was submitted
  twice through LLMConduit. The cold request completed in 48.691 seconds. The
  second request reused the 110,592-token recurrent checkpoint, recomputed
  1,709 tokens, and completed in 6.225 seconds. Both responses contained
  coherent reasoning, content, and a valid tool call; neither response emitted
  a Kimi control marker.
- A deterministic 500,224-token completion was submitted cold and then from a
  491,520-token recurrent checkpoint. The requests completed in 230.376 and
  8.394 seconds. All 16 sampled tokens and decoded text matched. The maximum
  absolute sampled-token log-probability difference was 0.000104.

## Uneven vision-shard collective

Data-parallel vision execution can assign an image to one tensor-parallel rank
while the other ranks receive zero encoder rows. Padding every rank to the
largest row count makes the subsequent all-gather allocate the largest image
once per rank. For one Kimi image with 1,792 encoder rows, merge factor four,
hidden width 1,024, BF16 activations, and TP16, the padded collective allocates
28,672 output rows, or 224 MiB per GPU.

[Local vLLM #464](https://github.com/local-inference-lab/vllm/pull/464)
gathers each rank's exact row count. The same TP16 shape allocates 1,792 output
rows, or 14 MiB per GPU. The gathered BF16 tensor matches the padded reference
bit-for-bit. The generic uniform-size path remains unchanged when all rank
sizes are equal.

The source change passed the PyNccl variable-size collective test with a
zero-row rank and five vision-model tests covering one image, three images,
five images, no images, and uneven four-rank image assignment. The full-model
qualification warmed four archived OMP image blobs and replayed the original
106-message continuation with those four images plus one new 1,408-by-960
image. LLMConduit returned HTTP 200 with 21 SSE events, a terminal done event,
no stream error, no Kimi control marker, and coherent image-dependent output.
The EngineCore remained healthy with 1,016,293 physical target KV tokens.

The replay receipt is recorded in
[`validation/kimi-k3-upstream-aligned-r29-20260821.json`](validation/kimi-k3-upstream-aligned-r29-20260821.json).

## Per-cache DCP topology

Each cache specification defines how many token-position shards it stores.
DCP-sharded target attention uses the configured DCP size. Recurrent state and
fully position-replicated speculative-draft caches use one token-position
shard, although tensor parallelism may still partition feature dimensions.

For the DFlash TP16/DCP16 profile, target MLA groups use 86 block-table columns,
recurrent groups use eight columns, and the replicated sliding-window draft
uses 1,366 columns. The contract prevents recurrent and draft tables from being
underallocated without changing physical KV capacity.

The Infernal Invocation adaptation is
[local vLLM #418](https://github.com/local-inference-lab/vllm/pull/418).
The corresponding official-vLLM implementation is maintained on branch
[`fix/dcp-cache-topology-contract-20260821`](https://github.com/voipmonitor/vllm/tree/fix/dcp-cache-topology-contract-20260821).

## Source composition

The Docker source lock composes these pull requests in order:

```text
vLLM: 414, 295, 294, 320, 413, 422, 310, 415, 418, 419, 459, 460, 463, 464
B12X: 227, 238
```

The exact PR heads, merge commits, bases, and trees are stored in:

```text
patches/releases/kimi-k3-upstream-aligned-20260821/vllm/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260821/b12x/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260821/lmcache/source.lock.json
```

## Rebuild

Check out the exact recipe commit to reproduce the published image metadata:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout bb3a4f954cfcc831dd4520a883a402eb09e66e62
./build-kimi-k3-vllm-python-refresh.sh
```

## Operational limits

- TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs is qualified. Other topologies
  require separate runtime measurement.
- One million token positions are allocation-qualified. The DSpark recurrent
  prefix-hit path is numerically qualified through 500,224 tokens. Broader
  semantic quality at that depth requires a separate evaluation.
- The production scheduler permits one active sequence.
- Each prompt may contain at most five images. Each image is bounded to 40,960
  input patches and 512 patches on one side.
- Native host KV offload is the qualified production cache backend. LMCache is
  available in the image but is not the low-latency production default.

The immutable rollback image is
`voipmonitor/vllm@sha256:e0a7d6f9f7e0ce7587d024520557b4b4399c04314623a7c5d78a3bf1882ecf71`.
