# Kimi-K3 MXFP4 Runtime

Status: **qualified** on 16 RTX PRO 6000 Blackwell GPUs.

The runtime serves the official `moonshotai/Kimi-K3` MXFP4 checkpoint in three
profiles:

- Inferact DSpark with native vision, prefix caching, and native vLLM host-KV
  offload;
- target-only decode without speculation;
- modal-labs DFlash speculative decode.

Inferact DSpark is the production profile. It exposes a 1,000,000-token request
limit and 1,016,293 physical FP8 target-KV tokens while retaining 32 GiB of host
RAM for native vLLM KV offload.

The machine-readable evidence is
[`validation/kimi-k3-upstream-aligned-r35-20260822.json`](validation/kimi-k3-upstream-aligned-r35-20260822.json).

## Immutable artifacts

| Component | Identity |
|---|---|
| vLLM image | `voipmonitor/vllm@sha256:e009bb404211c67164f1009bda97823f35578285b6779a7614ed1f97c1f8c338` |
| Image tag | `voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllme755f87-b12x2d466e3-cu133-torch213-20260822-r35` |
| Image ID | `sha256:f0240cfe8ab56b435d7ea4bea9a67479406ee5636357719a15db98e34836add5` |
| Docker recipe | `local-inference-lab/blackwell-llm-docker@daaaa0b` |
| Qualification record | `local-inference-lab/blackwell-llm-docker@026955c` |
| vLLM source tree | `e755f87b8e00d76e1aeacfa0835a2c7608925390` |
| B12X source tree | `2d466e350e518193f9edd57809e050b3aa8b8dcb` |
| FlashInfer wheels | `voipmonitor/vllm:flashinfer-wheels-fi1ac6942-cu133-torch213-20260820-r1@sha256:477a3b55b973df48b08a6dfae4a2a1e64c975a990dda22f65e31acd5217b86bb` |
| LLMConduit image | `voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61` |

The image uses CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL
4.6.2, FlashInfer 0.6.18, InstantTensor 0.1.9, and xgrammar 0.2.5. It contains
installed packages and does not mount or activate a source overlay.

## Model representation

| Property | Value |
|---|---|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Routed expert weights | checkpoint MXFP4 |
| Selected dense target projections | online MXFP8 weight-only representation |
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

## Start the DSpark production service

The Hugging Face cache must contain the pinned target and Inferact DSpark
snapshots. Do not set `NCCL_GRAPH_FILE` to an empty value.

```bash
IMAGE=voipmonitor/vllm@sha256:e009bb404211c67164f1009bda97823f35578285b6779a7614ed1f97c1f8c338
CACHE_DIR=/mnt/luke/kimi-k3-cache/kimi-k3-cu133-torch213

mkdir -p "$CACHE_DIR"
docker pull "$IMAGE"

docker run -d \
  --name kimi-k3-production-dspark \
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
  -e B12X_W4A16_STABLE_ROUTE_PACK=1 \
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
docker logs -f kimi-k3-production-dspark
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

Port 8001 is the vLLM API. The launcher loads the target with InstantTensor,
uses B12X MLA and routed-MoE kernels, and converts the supported target and
draft dense projections to MXFP8 before allocating the physical KV cache.

## Start the target-only service

The target-only profile is text-only and disables prefix caching and host-KV
offload. It provides 1,058,823 physical FP8 target-KV tokens.

```bash
docker run -d \
  --name kimi-k3-target-only \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE_DIR":/cache/jit:rw \
  -e PORT=8001 -e TP_SIZE=16 -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1000000 -e MAX_NUM_BATCHED_TOKENS=4102 \
  -e MAX_NUM_SEQS=1 -e KV_CACHE_MEMORY_BYTES=960000000 \
  -e ENABLE_PREFIX_CACHING=0 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e B12X_W4A16_STABLE_ROUTE_PACK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii \
  "$IMAGE" --max-num-scheduled-tokens 4096
```

## Start the DFlash service

The DFlash profile is text-only. It supplies the drafter with Kimi-K3's
pre-normalization AttnRes mixture and stages its six target auxiliary states
directly into one MXFP8 projection input. It provides 1,022,624 physical FP8
target-KV tokens.

```bash
docker run -d \
  --name kimi-k3-dflash \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "$CACHE_DIR":/cache/jit:rw \
  -e PORT=8001 -e TP_SIZE=16 -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1000000 -e MAX_NUM_BATCHED_TOKENS=4102 \
  -e MAX_NUM_SEQS=1 -e KV_CACHE_MEMORY_BYTES=1200000000 \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e B12X_W4A16_STABLE_ROUTE_PACK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii \
  "$IMAGE" --max-num-scheduled-tokens 4096
```

The server log must contain an activation record for an actual scheduler
forward, for example:

```text
DFlash staged auxiliary projection is active: tokens=4096 target_width=7168 slices=6.
```

Absence of that line means the bounded long-prefill path is not active.

## Qualification results

The normalized measurements use a pinned 256-token prompt, temperature zero,
and measured requests after warmup. Target cycles per second isolate target
execution speed from prompt-dependent speculative acceptance.

| Profile | Runs | Emitted median | Acceptance median | Target cycles/s median | Physical target-KV tokens |
|---|---:|---:|---:|---:|---:|
| Target-only | 3 | 55.807 tok/s | not applicable | not applicable | 1,058,823 |
| Inferact DSpark | 7 | 122.706 tok/s | 0.414940 | 31.426 | 1,016,293 |
| modal-labs DFlash | 7 | 155.341 tok/s | 0.618801 | 29.136 | 1,022,624 |

Output hashes for all three profiles match their source-locked controls.
Relative to the source composition without direct caller-owned DFlash output,
target-only changed by -0.008%, DSpark by +0.012%, and natural DFlash by
+4.969%.

Long-context evidence:

- The target-only and DSpark paths completed 208,026 input tokens plus 2,048
  generated tokens under the staged-input control composition. Their
  deterministic decode outputs in the published image are identical because
  the direct-output change is reachable only from DFlash auxiliary projection.
- The published DFlash path completed 524,288 input tokens plus 64 generated
  tokens with its staged direct-output path active.
- The published DSpark production profile passed reasoning, required tool-call,
  and native strawberry-image API checks.

Every streamed long-context response returned HTTP 200 with a terminal event,
no stream error, no Kimi protocol marker, and no repeated em-dash run.

Do not run `nvidia-smi dmon` during performance qualification. Concurrent NVML
polling has reproduced a persistent target-cycle reduction on an unchanged
control process.

## Bounded DFlash auxiliary projection

For 4,096 target tokens, six BF16 auxiliary states of width 7,168 concatenate
to a 336 MiB tensor. One-shot MXFP8 input quantization then retains a second
168 MiB tensor. B12X
[#241](https://github.com/local-inference-lab/b12x/pull/241) exposes retained
MXFP8 input allocation, aligned slice quantization, and GEMM from a prequantized
input. vLLM [#473](https://github.com/local-inference-lab/vllm/pull/473)
feeds each Kimi-K3 auxiliary state into that input as it is produced.

At `M=4096`, six `K=7168` slices, and `N=7168`, staged and concatenated
projection outputs are bitwise equal. Peak allocated memory falls from
1,278,083,584 to 663,355,904 bytes, a 48.10% reduction. Nine interleaved runs
measured 5.930 ms for concatenated input and 5.844 ms for staged direct output.
The final projection remains one GEMM with the original accumulation order.

## Native host-KV offload

The DSpark profile uses vLLM's native CPU offload backend, not LMCache. One
shared 32 GiB mmap region stores target attention and recurrent state across
all 16 ranks. Volatile draft-tail state is excluded from offload.

The unchanged native-cache composition has a qualified reset-and-restore
receipt at:

```text
/mnt/luke/kimi-k3-runs/r33-stable-route-source-locked-20260822/dspark/native-host-kv-repeat.json
```

The receipt records 1,221,083,136 CPU-to-GPU bytes across 16 rank transfers for
one restored prefix. The qualified image also recorded 22,722,674,688
GPU-to-CPU bytes while processing the DSpark regression suite.

## LLMConduit and Oh My Pi

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
```

Port 8003 is the LLMConduit API. The profile permits completion output up to
128,000 tokens, streams tool calls, preserves reasoning, and accepts native
image inputs.

Global OMP configuration:

```bash
mkdir -p /root/.omp/agent
cp models/kimi-k3/configs/omp-models.yml /root/.omp/agent/models.yml
cp models/kimi-k3/configs/omp-config.yml /root/.omp/agent/config.yml

omp models --json
omp --thinking high
```

Supported OMP reasoning values are `minimal`, `low`, `medium`, `high`, `xhigh`,
and `max`. OMP retains at most five images for this provider, matching the vLLM
request limit.

## Source composition

The Docker source lock composes these pull-request heads in order:

```text
vLLM: 414, 295, 294, 320, 413, 422, 310, 415, 418, 419, 459, 460,
      463, 464, 467, 468, 469, 471, then #473 at ee69840
B12X: 227, 238, 239, then #241 at bebd334
```

The vLLM staged auxiliary projection integration is published for review as
[#473](https://github.com/local-inference-lab/vllm/pull/473). Merge vLLM #460
before #473. The exact heads, merge commits, bases, and trees are stored in:

```text
patches/releases/kimi-k3-upstream-aligned-20260822/vllm/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260822/b12x/source.lock.json
patches/releases/kimi-k3-upstream-aligned-20260822/lmcache/source.lock.json
```

The maintainer merge order and official-vLLM disposition are maintained in
[rtx6kpro issue #75](https://github.com/local-inference-lab/rtx6kpro/issues/75).

## Rebuild

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout daaaa0b
./build-kimi-k3-upstream-aligned-runtime.sh
```

The build verifies every source-lock commit and tree before compiling. The
FlashInfer wheel image is immutable and independent of vLLM and B12X source
changes.

## Operational limits

- TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs is qualified. The generic B12X
  collectives used by the source composition support TP8 and TP12, but those
  Kimi-K3 topologies require separate full-model qualification.
- One million token positions are allocation-qualified. Long-context checks
  cover target-only and speculative execution as listed above; they do not
  establish semantic quality at every depth.
- The production scheduler permits one active sequence.
- DSpark permits at most five images per prompt. DFlash and target-only are
  qualified as text-only profiles.
- Native host-KV offload is the qualified production cache backend. LMCache is
  packaged but is not the low-latency production default.

The immutable rollback image is
`voipmonitor/vllm@sha256:e0a7d6f9f7e0ce7587d024520557b4b4399c04314623a7c5d78a3bf1882ecf71`.
