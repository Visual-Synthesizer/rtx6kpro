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
[`validation/kimi-k3-upstream-aligned-r26-20260821.json`](validation/kimi-k3-upstream-aligned-r26-20260821.json).

## Immutable artifacts

| Component | Identity |
|---|---|
| vLLM image | `voipmonitor/vllm@sha256:60ddcb1ebae94c21d66c8a0433952538c3a77feb6712eb2c907c0e727426c8b2` |
| Image tag | `voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllmddf87d6-b12xf006681-cu133-torch213-20260821-r26` |
| Image ID | `sha256:24675acf9976b770dce9d937e81fbce7621b15d305771d847f3fa4b2e6fb9f73` |
| Docker recipe | `local-inference-lab/blackwell-llm-docker@7b3cedb9d8a2a9f45624940e6e63c52b4fd6dcf1` |
| Docker `main` publication | `fb0ecfed677ea60c020f0d141c987aa5fd258ca8` |
| vLLM integration tree | `ddf87d676505d4e1c920357d4f9da2a58e2c8ec7` |
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
IMAGE=voipmonitor/vllm@sha256:60ddcb1ebae94c21d66c8a0433952538c3a77feb6712eb2c907c0e727426c8b2
CACHE_DIR=/mnt/luke/kimi-k3-cache/kimi-k3-r26

mkdir -p "$CACHE_DIR"
docker pull "$IMAGE"

docker run -d \
  --name kimi-k3-r26-production-dspark \
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
docker logs -f kimi-k3-r26-production-dspark
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
  --name kimi-k3-r26-nospec \
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
  --name kimi-k3-r26-dflash \
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
tokens, temperature zero, and seven measured requests after warmup.

| Profile | Emitted median | Acceptance median | Target cycles/s median | Physical KV tokens |
|---|---:|---:|---:|---:|
| Target-only | 55.693 tok/s | not applicable | not applicable | 1,458,823 |
| Inferact DSpark | 121.439 tok/s | 0.413961 | 31.417 | 1,016,293 |
| modal-labs DFlash | 139.187 tok/s | 0.533246 | 29.442 | 1,048,576 |

Speculative emitted throughput depends on generated token acceptance. Target
cycles per second are the stable hot-path regression metric. The DFlash target
rate differs by +0.10% from the isolated pre-normalization AttnRes reference;
the target-only rate differs by +0.02% from its 55.683 tok/s reference.

A three-run sanity test executed from the published image produced 31.526
target cycles/s median. Its emitted median was 147.941 tok/s at 0.527523 draft
acceptance; emitted throughput is not compared across prompts with different
acceptance.

Uncached DSpark prefill uses a unique cache salt and disables host offload for
each request:

| Prompt tokens | Median effective prefill |
|---:|---:|
| 8,192 | 3,641.10 tok/s |
| 32,768 | 3,693.89 tok/s |
| 65,535 | 3,531.30 tok/s |

A 134,209-token OMP replay containing five images completed both cold and on an
immediate repeat. Both requests returned HTTP 200, completed their streams,
emitted zero Kimi protocol markers, and left the engine healthy. The repeat
restored 122,880 prompt tokens through native host KV offload.

## Source composition

The Docker source lock composes these pull requests in order:

```text
vLLM: 414, 295, 294, 320, 413, 422, 310, 415, 418, 419, 459, 460
B12X: 227, 238
```

The exact PR heads, merge commits, bases, and trees are stored in:

```text
patches/releases/kimi-k3-upstream-aligned-20260821/vllm/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260821/b12x/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260821/lmcache/source.lock.json
```

## Rebuild

Checkout the exact recipe commit to reproduce the published image metadata:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 7b3cedb9d8a2a9f45624940e6e63c52b4fd6dcf1
./build-kimi-k3-vllm-python-refresh.sh
```

The same recipe content is present on `main` at
`fb0ecfed677ea60c020f0d141c987aa5fd258ca8`. Building from that merge commit
changes the OCI revision label and therefore the image digest, but not the
locked vLLM or B12X source trees.

## Operational limits

- TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs is qualified. Other topologies
  require separate runtime measurement.
- One million token positions are allocation- and replay-qualified. Model
  quality for coherent documents above 500,224 tokens is unsupported.
- The production scheduler permits one active sequence.
- Each prompt may contain at most five images. Each image is bounded to 40,960
  input patches and 512 patches on one side.
- Native host KV offload is the qualified production cache backend. LMCache is
  available in the image but is not the low-latency production default.
