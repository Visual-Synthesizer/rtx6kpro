# Kimi-K3 Official MXFP4 Production Deployment

Status: **qualified**.

This document specifies a reproducible Kimi-K3 service on 16 NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs. The service combines the official
`moonshotai/Kimi-K3` checkpoint, Inferact DSpark speculative decoding,
TP16/DCP16, native image input, structured tool calls, CPU-only LMCache,
LLMConduit, and Oh My Pi.

The machine-readable result for the runtime that includes recurrent-cache and
reasoning-stream corrections is
[`validation/production-dspark-lmcache-mamba-dcp-protocol-20260818.json`](validation/production-dspark-lmcache-mamba-dcp-protocol-20260818.json).

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
| Images per prompt | 2 |
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
| vLLM runtime image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllm452bd5c-b12xec6edd9-cu133-torch213-20260818-two-image` |
| vLLM runtime digest | `sha256:3aed14b70f54ea14c9d109e2df04c53df37db69d8182e430c43deba503c705e1` |
| vLLM source tree | `452bd5c56d7fb64de808d5a111c2272c70674c80` |
| B12X source tree | `ec6edd9da4687f83519fd37bd7322ea0800f0ace` |
| LMCache source tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Docker image recipe revision | `da2a8be7a7270609786ff1bb06c263c7197556f9` |
| LLMConduit image | `voipmonitor/llmconduit:kimi-k3-a628f0a-20260817-r4` |
| LLMConduit digest | `sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61` |
| LLMConduit source revision | `a628f0ae61b3362b8c3e571879d55d7ea36de5d2` |
| Long-context cache correction | [`vLLM#418`](https://github.com/local-inference-lab/vllm/pull/418), commit `6b18a8a767f406f08a519757ca6d5ef118b18296` |
| Kimi stream-parser correction | [`vLLM#419`](https://github.com/local-inference-lab/vllm/pull/419), commit `04a6acfe467f4a208c7231a18fc99faf656d016a` |

The source locks are stored at Docker recipe revision
`da2a8be7a7270609786ff1bb06c263c7197556f9` under
`patches/releases/kimi-k3-production-lmcache-mamba-dcp-protocol/`. Each lock records the
repository base, pull-request revisions, patch hash, and resulting Git tree.
The LLMConduit reasoning-control change is
[`llmconduit#37`](https://github.com/local-inference-lab/llmconduit/pull/37).
The maintainer merge map is
[`rtx6kpro#75`](https://github.com/local-inference-lab/rtx6kpro/issues/75).

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
  long-context language quality.

One million token positions are allocation-qualified. Natural or
model-generated inputs are qualified through 500,224 tokens; behavior from
500,225 through 1,000,000 coherent tokens remains unsupported until a coherent
input in that range is tested.

## Start vLLM, DSpark, Vision, and LMCache

The Hugging Face cache must contain both pinned checkpoint snapshots at the
paths used by the launcher. Pulling by digest provides the byte-identical
qualified image.

```bash
docker pull \
  voipmonitor/vllm@sha256:3aed14b70f54ea14c9d109e2df04c53df37db69d8182e430c43deba503c705e1

mkdir -p /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-452bd5c-ec6edd9

docker run -d \
  --name kimi-k3-production-mamba-dcp-protocol \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --security-opt label=disable \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-452bd5c-ec6edd9:/cache/jit:rw \
  voipmonitor/vllm@sha256:3aed14b70f54ea14c9d109e2df04c53df37db69d8182e430c43deba503c705e1
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
docker logs -f kimi-k3-production-mamba-dcp-protocol
curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8100/healthcheck
```

Startup on the qualification host read 1.41 TiB through InstantTensor in
approximately 160 seconds. Complete model loading took 179.3 seconds and allocated
90.48 GiB per GPU. CUDA graph capture took 48 seconds and 0.29 GiB per GPU.

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

The normalized decode protocol used one request, 256 stored input tokens,
1,024 generated tokens, greedy sampling, one seed, two unrecorded warmups, and
eight measured runs.

| Metric | Median |
|---|---:|
| Emitted decode throughput | 113.71 tok/s |
| Target cycles | 31.394 cycles/s |
| Emitted tokens per target cycle | 3.624 |
| Draft acceptance | 0.3749 |

Emitted throughput varies with draft acceptance. Target-cycle throughput is
the acceptance-independent runtime regression control. The image without the
recurrent-cache and stream-parser corrections measured 31.430 target cycles/s
under the same protocol. The corrected image measures 31.394 target cycles/s,
a 0.11% reduction that is below run-to-run variation.

The Kimi XTML stream parser was also tested by replaying a captured OMP turn
through LLMConduit. The request completed with 124 SSE events and 28,629 bytes,
and emitted zero `<|open|>`, `<|close|>`, or `<|sep|>` markers. LLMConduit turn
captures replace image URLs and data URIs with `<redacted uri>` so diagnostic
artifacts do not retain image payloads or credentials. The replay tool
[`tools/replay-llmconduit-turn-capture.py`](tools/replay-llmconduit-turn-capture.py)
removes those placeholders by default, accepts explicit replacements through
`--replacement-image-url`, and can reject incomplete replay input through
`--fail-on-redacted-images`.

The two-image prompt limit was qualified through LLMConduit with two distinct
native image payloads and through Oh My Pi with the failure-sensitive agent
sequence: the first turn attached one image, and a continued second turn sent
the archived image together with a newly attached image. The second backend
request contained two image items, completed normally, and did not produce the
repeated HTTP 400 failure caused by a one-image runtime limit. The qualified
captures are
`api_c6bc9d693d5b4618baad13487e76e497.json` for the first turn and
`api_b76fbb72f3d349aa884a48a4102f86dd.json` for the continued turn under
`/mnt/luke/kimi-k3-runs/llmconduit-turn-captures/`.

Replay a redacted capture without image content:

```bash
python3 models/kimi-k3/tools/replay-llmconduit-turn-capture.py \
  /path/to/capture.json \
  --max-completion-tokens 256 \
  --output /tmp/replay.sse \
  --receipt /tmp/replay.json
```

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
git checkout da2a8be7a7270609786ff1bb06c263c7197556f9

IMAGE=voipmonitor/vllm:kimi-k3-production-local \
RELEASE_DATE=20260818 \
REVISION=two-image \
./build-kimi-k3-qsrt-tp16-runtime.sh
```

The builder starts from the pinned CUDA 13.3 and PyTorch 2.13 base image,
checks all three integration locks, builds native extensions against the same
ABI, validates required launchers and imports, and runs source-composition
tests. Compiler metadata can produce a different Docker digest even when the
verified source trees match. The vLLM lock composes PRs #310, #413, #414,
#415, #418, and #419 into tree
`452bd5c56d7fb64de808d5a111c2272c70674c80`.

## Operational Limits

- The qualified topology is TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs.
- Natural or model-generated inputs are qualified through 500,224 tokens.
  One million token positions are allocation-qualified. Coherent language
  behavior above 500,224 tokens has not been qualified.
- The scheduler permits one active sequence. Parallel request throughput has
  not been qualified for the vision-and-LMCache allocation.
- Two images are permitted per prompt. This supports an agent turn containing
  one archived frame and one newly attached image. Image preprocessing is
  bounded to 40,960 input patches and 512 patches on one side.
- The qualified LMCache tier is CPU RAM only. The launcher implements a
  filesystem tier, but that tier is not part of this qualification.
- A host-local proxy exposes port 8000 on the qualification machine. The vLLM
  container binds port 8001; port 8000 is not created by the image.
- LLMConduit rewrites a `developer` role to `system` for the Kimi template and
  rejects unrecognized roles.
