# Kimi-K3 Official MXFP4 Production Deployment

Status: **qualified**.

This document specifies a reproducible Kimi-K3 service on 16 NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs. The service combines the official
`moonshotai/Kimi-K3` checkpoint, Inferact DSpark speculative decoding,
TP16/DCP16, native image input, structured tool calls, CPU-only LMCache,
LLMConduit, and Oh My Pi.

The machine-readable qualification receipt for the source-activated runtime is
[`validation/production-dspark-lmcache-source-overlay-20260819.json`](validation/production-dspark-lmcache-source-overlay-20260819.json).

## Deployment Contract

| Property | Qualified value |
|---|---|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Draft checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| Topology | TP16/DCP16 on 16 GPUs |
| Served model | `Kimi-K3-MXFP4-DSpark7-DCP16-1M` |
| Target KV cache | FP8, 1,033,126 physical tokens |
| Configured maximum request length | 1,000,000 tokens |
| Longest model-qualified prompt with vLLM #418 | 500,224 tokens |
| Long-prefill cache-index stress prompt | 520,002 tokens plus 64 generated tokens |
| Images per prompt | 5 |
| Speculative depth | 7 draft tokens |
| vLLM endpoint | `http://127.0.0.1:8001/v1` |
| LMCache control and metrics | `http://127.0.0.1:8100` |
| LLMConduit endpoint | `http://127.0.0.1:8003` |
| CUDA / PyTorch / NCCL | 13.3 / 2.13.0 / 2.31.2 |
| Weight loader | InstantTensor 0.1.9 |

The target is not a three-bit hybrid checkpoint. Routed experts remain in the
official checkpoint's MXFP4 representation. At load time, weight-only MXFP8 is
applied to the target KDA `q_proj`, `k_proj`, `v_proj`, `b_proj`, and `f_a_proj`
linears and to vision-tower and multimodal-projector linears. Activations remain
BF16. The DSpark draft uses weight-only MXFP8 except for
`fused_qkv_a_proj`, which remains BF16. The target and draft KV caches use FP8.

Native vision is part of the qualified allocation. Vision and projector
weights are loaded before deferred target tensors and remain resident on the
GPUs after online MXFP8 conversion. The physical KV-cache measurement of
1,033,126 tokens therefore already includes vision memory.

## Immutable Artifacts

| Artifact | Identifier |
|---|---|
| vLLM runtime image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllm12776c0-b12x468c312-cu133-torch213-20260819-source-overlay` |
| vLLM runtime digest | `sha256:a54da4e2432138d42334cac54555f3a51188489cb66029da0d96e4b39162d726` |
| vLLM source tree | `12776c0df15ca4087b636c43004b5bc1fde61434` |
| B12X source tree | `468c31256b585c8078782c63ba723e404c04eb76` |
| LMCache source tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Docker image recipe revision | `f9cc49c187f7525453b65ed8b4a1733f6d97471a` |
| LLMConduit image | `voipmonitor/llmconduit:kimi-k3-a628f0a-20260817-r4` |
| LLMConduit digest | `sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61` |
| LLMConduit source revision | `a628f0ae61b3362b8c3e571879d55d7ea36de5d2` |
| Oh My Pi client | `v17.3.5`, commit `37eee71978951fccf66b21f7e3e2b74596ac9d74` |
| Long-context cache correction | [`vLLM#418`](https://github.com/local-inference-lab/vllm/pull/418), commit `6b18a8a767f406f08a519757ca6d5ef118b18296` |
| Kimi stream-parser correction | [`vLLM#419`](https://github.com/local-inference-lab/vllm/pull/419), commit `04a6acfe467f4a208c7231a18fc99faf656d016a` |
| Hybrid KV-load failure isolation | [`vLLM#422`](https://github.com/local-inference-lab/vllm/pull/422), commit `6bafa633153a44ba1aa54eb7c5bafac248fe68e6` |
| Interleaved MLA query materialization | [`vLLM#427`](https://github.com/local-inference-lab/vllm/pull/427), commit `0ad95065da12e45c28d585d8c758a87d3a708b1f` |
| DCP hybrid local-prefix correction | [`vLLM#401`](https://github.com/local-inference-lab/vllm/pull/401), merge commit `f8390f93a8057833803f866dcddf619c7606ecde` |
| Cached logits-processing state | [`vLLM#433`](https://github.com/local-inference-lab/vllm/pull/433), commit `9151d114e270250fb0367333b2dc5a49c6383796` |
| W4A16 inactive-route handling | [`B12X#227`](https://github.com/local-inference-lab/b12x/pull/227), commit `0eba6ae99e0d1fad6ec268d8c291f498ec1dd4d9` |

The source locks are stored at Docker recipe revision
`f9cc49c187f7525453b65ed8b4a1733f6d97471a` under
`patches/releases/kimi-k3-production-lmcache-sampler-overlay/`. Each lock records the
repository base, pull-request revisions, patch hash, and resulting Git tree.
The LLMConduit reasoning-control change is
[`llmconduit#37`](https://github.com/local-inference-lab/llmconduit/pull/37).
The maintainer merge map is
[`rtx6kpro#75`](https://github.com/local-inference-lab/rtx6kpro/issues/75).

## Hybrid KV-Load Failure Contract

Kimi-K3 uses 17 KV-cache groups: 16 full-attention groups and one recurrent
sliding-window group. A failed external-cache load can therefore invalidate a
different set of blocks in each group. vLLM #422 evaluates all cache groups,
uses each group's block size for token alignment, and applies the configured
failure policy without assuming that a request has only one group.

The production launcher uses `kv_load_failure_policy=fail`. A failed LMCache
load terminates only the affected request with `FINISHED_ERROR`; the engine
continues scheduling unrelated requests. The `recompute` policy is unsupported
for distributed external-cache loads until connectors report peer-rank block
availability. Recomputing from rank-local metadata can otherwise reuse stale
blocks on another rank.

Validation used the connector scheduler suite and a Kimi-shaped regression:

- 20 external-cache connector tests passed;
- seven selected hybrid-cache and Mamba scheduler tests passed;
- a 17-group request with an invalid block in the recurrent group finished
  with `FINISHED_ERROR`, and a subsequent healthy request scheduled normally;
- the same 17-group input raises `ValueError: too many values to unpack` when
  executed against the single-group implementation.

## Interleaved MLA Query Contract

Kimi-K3 forms the absorbed MLA NoPE query as a token-major tensor whose local
head rows are interleaved in storage. For a 14-token TP16 request, the tensor
has shape `(14, 6, 128)` and stride `(1152, 192, 1)`. Transposing tokens and
heads produces a non-contiguous `(6, 14, 128)` view. Passing that view directly
to the cuBLAS batched query-times-`W_UV` operation permits vectorized reads to
cross the backing allocation boundary. The observed read crossed the boundary
by 4,608 bytes and produced `CUBLAS_STATUS_INTERNAL_ERROR` when LMCache selected
PyTorch's native CUDA allocator.

vLLM #427 materializes the head-major query view before the batched matrix
multiplication. It preserves BF16 arithmetic, output shape, and decode graph
semantics. It does not add a CUDA extension or change the KDA projection
quantization.

The immutable image identified above completed the exact 14-token cold
reproducer and 20 unique cold variants: 21 of 21 requests returned HTTP 200,
each generated four tokens, and the server reported no Xid, cuBLAS, illegal
memory access, or nonfinite-output errors.

## Long-Context Recurrent-Cache Contract

Under DCP16, recurrent KDA state has one token-position shard across the DCP
group. `MambaManager` therefore indexes state at absolute 768-token boundaries.
Treating that state as 16 position shards produces a block table that is too
narrow: a no-weight cache-topology reproducer reaches the first invalid table
column at 63,744 tokens. The resulting invalid physical block ID can address an
MLA cache page and make target logits nonfinite.

vLLM #418 defines recurrent state as one DCP token-position shard. TP may still
partition state feature dimensions. The correction does not change attention
DCP sharding, model weights, physical KV-cache allocation, or checkpoint
quantization. It adds approximately 58.6 KiB of worker block-table storage per
rank for a 1,000,000-token profile.

The runtime image identified above includes vLLM #418. Qualification produced
these results:

- the no-weight 17-group cache reproducer completed 1,000,000 tokens in 1,302
  allocation steps with unique physical-page ownership;
- official Kimi-K3 MXFP4 with Inferact DSpark K7 on TP16/DCP16 completed a
  500,224-token model-generated prefill in 1,258.884 seconds with finite output;
- a subsequent 128-token decode on the same process produced 640 finite
  top-logprob values and zero nonfinite values;
- the immutable production image completed an exact captured 140,960-token
  prompt with 21 of 21 finite logprob values in 106.196 seconds;
- target-only execution completed a 520,000-token cache-index stress input with
  21 of 21 finite logprob values in 464.812 seconds. That input repeats a
  captured 140,960-token sequence and validates cache indexing, not coherent
  long-context language quality;
- the source-locked DSpark and LMCache image completed a 520,002-token prompt
  plus 64 generated tokens in 648.194 seconds with coherent repeated-text
  output and no CUDA error markers. This validates the packaged speculative
  runtime and cache indexing, not coherent-document language quality.

One million token positions are allocation-qualified. Natural or
model-generated inputs are qualified through 500,224 tokens; behavior from
500,225 through 1,000,000 coherent tokens remains unsupported until a coherent
input in that range is tested.

## DCP Hybrid Local Prefix Cache

vLLM #401 permits a fine-grained local prefix hit only when every aligned
recurrent-cache manager has materialized a complete state at the candidate hash
boundary. A partial recurrent state retains the coarser DCP attention-block
boundary. External LMCache object geometry remains 12,288 tokens and is not
changed by this correction.

The TP16/DCP16 production service was tested with a 44,449-token producer and a
44,609-token consumer sharing the producer prefix. The producer took 30.919
seconds. The consumer reused 43,008 local tokens and completed in 1.741 seconds.
The external-prefix hit counter remained zero, proving that this result came
from vLLM's same-process local cache rather than LMCache.

The image sets `VLLM_SOURCE_OVERLAY_ACTIVE=1` globally and activates the pinned
source lazily. Both the packaged entrypoint and a custom Python or vLLM command
therefore import `/opt/kimi-k3-qsrt/vllm`; compiled extension modules fall back
to `/opt/infernal-invocation/vllm/vllm`. A custom command does not need to set a
source-overlay environment variable.

## Start vLLM, DSpark, Vision, and LMCache

The Hugging Face cache must contain both pinned checkpoint snapshots at the
paths used by the launcher. Pulling by digest provides the byte-identical
qualified image.

```bash
docker pull \
  voipmonitor/vllm@sha256:a54da4e2432138d42334cac54555f3a51188489cb66029da0d96e4b39162d726

mkdir -p /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-12776c0-468c312

docker run -d \
  --name kimi-k3-production-dspark-lmcache-12776c0 \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --security-opt label=disable \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-12776c0-468c312:/cache/jit:rw \
  voipmonitor/vllm@sha256:a54da4e2432138d42334cac54555f3a51188489cb66029da0d96e4b39162d726
```

The image entrypoint is
`/usr/local/bin/serve-kimi-k3-production-dspark-ii`. It starts one LMCache
server without CUDA visibility and then starts the 16 vLLM workers. LMCache
uses engine-driven GPU transfers: the existing vLLM workers gather and scatter
cache pages, while the standalone cache process owns only CPU memory. The
qualified RAM tier has a 32 GiB limit, zero initial allocation, 12,288-token
objects, and no disk tier.

Wait for readiness and verify the reported model:

```bash
docker logs -f kimi-k3-production-dspark-lmcache-12776c0
curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8100/healthcheck
```

Startup on the qualification host read 1.41 TiB through InstantTensor in
approximately 160 seconds. Target weight loading took 164.32 seconds, and draft
weight loading took 1.60 seconds.
All 16 workers compiled the bounded vision interpolation before allocation of
the 1,033,126-token KV cache. CUDA graph capture took approximately 51 seconds
and 0.29 GiB per GPU. LMCache uses PyTorch's native CUDA allocator; the wrapper
disables expandable segments before starting the vLLM workers because CUDA IPC
requires stable native allocations.

## Start LLMConduit

Copy [`configs/llmconduit-production.yaml`](configs/llmconduit-production.yaml)
to a stable host path. The qualified host uses
`/root/vllm/kimi/llmconduit-kimi-k3-production.yaml`.

```bash
docker pull \
  voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61

docker run -d \
  --name llmconduit-kimi-k3-production \
  --restart unless-stopped \
  --network host \
  -e LLMCONDUIT_BIND_ADDR=0.0.0.0:8003 \
  -e RUST_LOG=info \
  -v /root/vllm/kimi/llmconduit-kimi-k3-production.yaml:/config/config.yaml:ro \
  voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61 \
  start --config /config/config.yaml

curl -fsS http://127.0.0.1:8003/health
curl -fsS http://127.0.0.1:8003/v1/models | jq .
```

LLMConduit preserves Kimi reasoning internally for parser correctness. Client
controls determine whether parsed reasoning is returned:

| Requested level | Kimi backend level | Returned reasoning |
|---|---|---|
| `none` | internal thinking remains enabled | suppressed |
| `minimal`, `low`, `medium`, `high` | `high` | preserved |
| `xhigh`, `max` | `max` | preserved |
| omitted | `high` | preserved |

The mapping is qualified for OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages, and streaming responses. Required tool calls and native
image payloads pass through without flattening.

An OpenAI Chat Completions request can select the returned reasoning channel:

```bash
curl -fsS http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K3-MXFP4-DSpark7-DCP16-1M",
    "messages": [{"role": "user", "content": "Return the number after 41."}],
    "reasoning_effort": "none",
    "max_tokens": 64
  }' | jq .
```

Replace `none` with `high` or `max` to preserve `reasoning_content` in the
response.

## Configure Oh My Pi

Status: **qualified** with `omp/17.3.5`.

Install the two repository configurations as the Oh My Pi agent configuration:

```bash
mkdir -p /root/.omp/agent
cp models/kimi-k3/configs/omp-models.yml /root/.omp/agent/models.yml
cp models/kimi-k3/configs/omp-config.yml /root/.omp/agent/config.yml

omp models --json
```

The alias `production-mxfp4-dspark-dcp16` intentionally uses OMP's Z.AI
thinking wire format. It sends an explicit disabled-thinking object for
`--thinking off` and sends `reasoning_effort` for enabled levels.

```bash
omp --thinking off --no-session --no-tools --no-extensions --no-skills \
  --no-rules -p 'Reply with exactly READY.'

omp --thinking high --print-thoughts --no-session --no-tools \
  --no-extensions --no-skills --no-rules \
  -p 'Explain why 2 + 2 equals 4.'

omp --thinking high --print-thoughts --no-session --no-tools \
  --no-extensions --no-skills --no-rules \
  -p @/path/to/image.png 'Describe the supplied image.'
```

Qualification verified model discovery, hidden reasoning at `off`, preserved
reasoning at `high` and `max`, a two-turn file-read tool loop, native PNG
vision, and the exact wire controls emitted by OMP.

## Measured Behavior

The source-activated image was measured with one streamed coding request, 119
prompt tokens, 512 generated tokens, greedy sampling, and three runs. Throughput
uses the interval from the first streamed output token through the last; HTTP
time-to-first-token is excluded.

| Run | Active decode throughput |
|---:|---:|
| 1 | 133.55 tok/s |
| 2 | 118.72 tok/s |
| 3 | 91.05 tok/s |
| Median | 118.72 tok/s |

The comparison image measured 122.18 tok/s median under the same three-run
protocol. DSpark throughput varies with draft acceptance; the two medians are
inside the observed per-run range.

Docker digest
`sha256:5cb7978445e6f3bb3efd70102bc5737e4fc6f0bb6a89d3ede4b74882a5da899a`
completed three additional 1,024-token decode runs at 106.62, 121.91, and
108.28 emitted tok/s; the median was 108.28 tok/s. The qualified image in this
document completed one 512-token coding decode at 108.82 tok/s and three
unique uncached repeats at 114.68, 123.82, and 112.21 tok/s. The uncached-repeat
median is 114.68 tok/s. DSpark acceptance varies with generated content, so
target-cycle throughput from the normalized protocol remains the primary
language-runtime regression control.

The Kimi XTML stream parser was also tested by replaying a captured OMP turn
through LLMConduit. The request completed with 124 SSE events and 28,629 bytes,
and emitted zero `<|open|>`, `<|close|>`, or `<|sep|>` markers. LLMConduit turn
captures replace image URLs and data URIs with `<redacted uri>` so diagnostic
artifacts do not retain image payloads or credentials. The replay tool
[`tools/replay-llmconduit-turn-capture.py`](tools/replay-llmconduit-turn-capture.py)
removes those placeholders by default, accepts explicit replacements through
`--replacement-image-url`, and can reject incomplete replay input through
`--fail-on-redacted-images`.

OMP 17.3.5 retains at most five images for a custom provider and removes the
oldest transient images above that budget. The vLLM prompt limit is therefore
five so OMP can replay every image it is permitted to retain. A direct
LLMConduit request containing five WebP payloads completed with HTTP 200 and
returned the expected image count.

The request that exposed the hybrid-group scheduler defect contained 186
messages and five archived images. Replaying that exact message history with
five valid WebP data URIs through LLMConduit completed with HTTP 200 after
171.84 seconds. The model consumed 179,279 prompt tokens, emitted 865 tokens,
finished with a valid `write` tool call, delivered 672 SSE events plus
`[DONE]`, and emitted zero `<|open|>`, `<|close|>`, or `<|sep|>` markers. The
successful replay receipt is `api_7c37481cb259410192fd14b33009ed6a.json`; its
redacted input source is `api_23bd0f237db245c3a86ef522ec102e88.json`. The
receipts are stored under
`/mnt/luke/kimi-k3-runs/llmconduit-turn-captures/`.

The interleaved-query image also replayed captured OMP turn
`api_23ce09b17db24c4bbf3996e2c8f46de8.json` through LLMConduit. The model
consumed 60,922 prompt tokens, returned HTTP 200, emitted 45 SSE events plus
`[DONE]`, and finished with a syntactically valid `bash` tool call. The stream
contained no gateway error event and no `<|open|>`, `<|close|>`, or `<|sep|>`
marker. The receipt and raw stream are stored under
`/mnt/luke/kimi-k3-runs/interleaved-query-image-qualification-20260818/`.

Replay a redacted capture without image content:

```bash
python3 models/kimi-k3/tools/replay-llmconduit-turn-capture.py \
  /path/to/capture.json \
  --max-completion-tokens 256 \
  --output /tmp/replay.sse \
  --receipt /tmp/replay.json
```

The replay command exits nonzero for HTTP failures, missing `[DONE]`, XTML
control-marker leakage, and gateway error objects embedded in an otherwise
HTTP-200 SSE stream. A short generation that ends normally with
`finish_reason=length` is not classified as a transport failure.

For a multimodal replay, pass one `--replacement-image-url` for each captured
image in encounter order. A replacement may be an HTTP URL or a data URI. Use
`--fail-on-redacted-images` when dropping image content would invalidate the
test objective.

External LMCache restore was tested after clearing the vLLM-local prefix cache.
A 24,576-token prompt restored 12,288 tokens from LMCache. A 49,152-token
prompt restored 36,864 tokens. Both restored requests produced the same output
token as their cold controls. The live process inventory contained the 16 vLLM
GPU workers and no LMCache GPU process.

The direct vLLM API passed Kimi reasoning parsing, a required calculator tool
call, and native vision. LLMConduit passed Chat, Responses, Anthropic,
streaming, tool, and vision qualification. Oh My Pi passed reasoning, tool, and
vision qualification through LLMConduit.

Host-local raw receipts are stored under:

```text
/mnt/luke/kimi-k3-runs/qualification-mamba-dcp-protocol-20260818
/mnt/luke/kimi-k3-runs/reasoning-protocol-fix-20260818
/mnt/luke/kimi-k3-runs/interleaved-query-image-qualification-20260818
```

## Comparison Entry Points

The runtime image also contains these entrypoints:

| Entrypoint | Behavior | Status in the immutable production image |
|---|---|---|
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii` | Official target without speculative decoding | qualified at 56.149 tok/s with 1,460,937 physical KV tokens; language-only and without LMCache |
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii` | Official target with Inferact DSpark | qualified at 118.773 emitted tok/s; the production wrapper adds vision and LMCache |
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii` | Official target with modal-labs DFlash | qualified at 90.962 emitted tok/s with 1,048,576 physical KV tokens; language-only and without LMCache |

Only `/usr/local/bin/serve-kimi-k3-production-dspark-ii` has the complete
vision, LMCache, LLMConduit, and Oh My Pi qualification described by this
document. The comparison entrypoints must use separate writable `/cache/jit`
directories because their CUDA graph shapes differ.

## Rebuild the Runtime Image

Pulling by digest is required for byte identity. Rebuilding uses the release
locks and verifies the resulting source trees before compiling:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout f9cc49c187f7525453b65ed8b4a1733f6d97471a

IMAGE=voipmonitor/vllm:kimi-k3-production-local \
RELEASE_DATE=20260819 \
REVISION=source-overlay \
./build-kimi-k3-qsrt-tp16-runtime.sh
```

The builder starts from the pinned CUDA 13.3 and PyTorch 2.13 base image,
checks all three integration locks, builds native extensions against the same
ABI, validates required launchers and imports, and runs source-composition
tests. Compiler metadata can produce a different Docker digest even when the
verified source trees match. The vLLM base contains merged PR #401. The vLLM
lock composes PRs #310, #413, #414, #415, #418, #419, #422, #427, #428, and
#433 into tree `12776c0df15ca4087b636c43004b5bc1fde61434`. The B12X lock
composes PR #227 into tree `468c31256b585c8078782c63ba723e404c04eb76`.

## Operational Limits

- The qualified topology is TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs.
- Natural or model-generated inputs are qualified through 500,224 tokens.
  A 520,002-token repeated-text prompt is runtime-qualified. One million token
  positions are allocation-qualified. Coherent-document language behavior
  above 500,224 tokens has not been qualified.
- The scheduler permits one active sequence. Parallel request throughput has
  not been qualified for the vision-and-LMCache allocation.
- Five images are permitted per prompt, matching the OMP 17.3.5 custom-provider
  image budget. Image preprocessing is bounded to 40,960 input patches and 512
  patches on one side.
- The qualified LMCache tier is CPU RAM only. The launcher implements a
  filesystem tier, but that tier is not part of this qualification.
- External-cache load failures use the `fail` policy and terminate only the
  affected request. Distributed `recompute` is unsupported until every rank's
  external-cache block availability is available to the scheduler.
- A host-local proxy exposes port 8000 on the qualification machine. The vLLM
  container binds port 8001; port 8000 is not created by the image.
- LLMConduit rewrites a `developer` role to `system` for the Kimi template and
  rejects unrecognized roles.
