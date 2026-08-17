# Kimi-K3 Official MXFP4 Production Deployment

Status: **qualified for short-context behavior in the immutable image;
long-context use requires vLLM #418**.

This document specifies a reproducible Kimi-K3 service on 16 NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs. The service combines the official
`moonshotai/Kimi-K3` checkpoint, Inferact DSpark speculative decoding,
TP16/DCP16, native image input, structured tool calls, CPU-only LMCache,
LLMConduit, and Oh My Pi.

The machine-readable result is
[`validation/production-dspark-lmcache-20260817.json`](validation/production-dspark-lmcache-20260817.json).

## Deployment Contract

| Property | Qualified value |
|---|---|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Draft checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| Topology | TP16/DCP16 on 16 GPUs |
| Served model | `Kimi-K3-MXFP4-DSpark7-DCP16-1M` |
| Target KV cache | FP8, 1,033,126 physical tokens |
| Configured maximum request length | 1,000,000 tokens |
| Longest qualified prompt with vLLM #418 | 500,224 tokens |
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
| vLLM runtime image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-vllmdf13924-b12xec6edd9-cu133-torch213-20260817-r6` |
| vLLM runtime digest | `sha256:ffb25774eaa90850b4cacfb88ed9e55072818e99bad977f1315c7118e7a730b2` |
| vLLM source tree | `df13924cb12279f8144019800fad7e27640acaea` |
| B12X source tree | `ec6edd9da4687f83519fd37bd7322ea0800f0ace` |
| LMCache source tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Docker image recipe revision | `2cbc2e58e094ffbbd517afeec1619b391b998438` |
| Docker recipe merge revision | `26642da52049a41a25e425c630f94de113ea9e6a` |
| LLMConduit image | `voipmonitor/llmconduit:kimi-k3-5e07aec-20260817-r3` |
| LLMConduit digest | `sha256:a0d7416ebaed984fea33646b57a54d80b250a2f4e1257e08cfc4c07cb6699c7d` |
| LLMConduit source revision | `5e07aec44f48d8ac5ac64749ce1884083631fb5f` |
| Long-context cache correction | [`vLLM#418`](https://github.com/local-inference-lab/vllm/pull/418), commit `6b18a8a767f406f08a519757ca6d5ef118b18296` |

The source locks are stored in
[`blackwell-llm-docker#22`](https://github.com/local-inference-lab/blackwell-llm-docker/pull/22)
under `patches/releases/kimi-k3-production-lmcache-r6/`. Each lock records the
repository base, pull-request revisions, patch hash, and resulting Git tree.
The LLMConduit reasoning-control change is
[`llmconduit#37`](https://github.com/local-inference-lab/llmconduit/pull/37).
The maintainer merge map is
[`rtx6kpro#75`](https://github.com/local-inference-lab/rtx6kpro/issues/75).

## Long-Context Cache Requirement

The immutable runtime digest in this document does not contain vLLM #418.
Under DCP16 it sizes recurrent KDA block tables as if token positions were
sharded across 16 ranks, while `MambaManager` indexes recurrent state at
absolute 768-token boundaries. A no-weight cache-topology reproducer reaches
the first invalid table column at 63,744 tokens. The resulting invalid physical
block ID can address an MLA cache page and make target logits nonfinite.

vLLM #418 defines recurrent state as one DCP token-position shard. TP may still
partition state feature dimensions. The correction does not change attention
DCP sharding, model weights, physical KV-cache allocation, or checkpoint
quantization. It adds approximately 58.6 KiB of worker block-table storage per
rank for a 1,000,000-token profile.

Qualification with the vLLM #418 source overlay produced these results:

- the no-weight 17-group cache reproducer completed 1,000,000 tokens in 1,302
  allocation steps with unique physical-page ownership;
- official Kimi-K3 MXFP4 with Inferact DSpark K7 on TP16/DCP16 completed a
  500,224-token prefill in 1,258.884 seconds with finite output;
- a subsequent 128-token decode on the same process produced 640 finite
  top-logprob values and zero nonfinite values.

Do not use the immutable image by digest for prompts of 63,744 tokens or more.
The running qualification container uses the vLLM #418 source overlay. A
replacement immutable image must incorporate #418 before it can claim the
configured 1,000,000-token request length.

## Start vLLM, DSpark, Vision, and LMCache

The Hugging Face cache must contain both pinned checkpoint snapshots at the
paths used by the launcher. Pulling by digest provides the byte-identical
short-context image; it does not include the long-context correction described
above.

```bash
docker pull \
  voipmonitor/vllm@sha256:ffb25774eaa90850b4cacfb88ed9e55072818e99bad977f1315c7118e7a730b2

mkdir -p /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-df13924

docker run -d \
  --name kimi-k3-production-dspark-lmcache-df13924 \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --security-opt label=disable \
  -e LMCACHE_L1_INIT_GB=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e INSTANTTENSOR_BUFFER_SIZE=8388608 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/kimi-k3-production-lmcache-df13924:/cache/jit:rw \
  voipmonitor/vllm@sha256:ffb25774eaa90850b4cacfb88ed9e55072818e99bad977f1315c7118e7a730b2
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
docker logs -f kimi-k3-production-dspark-lmcache-df13924
curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8100/healthcheck
```

Startup on the qualification host read the target weights in 154.33 seconds.
Complete model loading took approximately 168.5 seconds and allocated
90.48 GiB per GPU. CUDA graph capture took 48 seconds and 0.29 GiB per GPU.

## Start LLMConduit

Copy [`configs/llmconduit-production.yaml`](configs/llmconduit-production.yaml)
to a stable host path. The qualified host uses
`/root/vllm/kimi/llmconduit-kimi-k3-production.yaml`.

```bash
docker pull \
  voipmonitor/llmconduit@sha256:a0d7416ebaed984fea33646b57a54d80b250a2f4e1257e08cfc4c07cb6699c7d

docker run -d \
  --name llmconduit-kimi-k3-production \
  --restart unless-stopped \
  --network host \
  -e LLMCONDUIT_BIND_ADDR=0.0.0.0:8003 \
  -e RUST_LOG=info \
  -v /root/vllm/kimi/llmconduit-kimi-k3-production.yaml:/config/config.yaml:ro \
  voipmonitor/llmconduit@sha256:a0d7416ebaed984fea33646b57a54d80b250a2f4e1257e08cfc4c07cb6699c7d \
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
| Emitted decode throughput | 118.77 tok/s |
| Target cycles | 31.43 cycles/s |
| Emitted tokens per target cycle | 3.781 |
| Draft acceptance | 0.3973 |

Emitted throughput varies with draft acceptance. Target-cycle throughput is
the acceptance-independent runtime regression control; the qualified image
measures 31.43 target cycles/s.

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
/mnt/luke/kimi-k3-runs/qualification-df13924-ec6edd9-20260817
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
git checkout 26642da52049a41a25e425c630f94de113ea9e6a

IMAGE=voipmonitor/vllm:kimi-k3-production-local \
RELEASE_DATE=20260817 \
REVISION=r6 \
./build-kimi-k3-qsrt-tp16-runtime.sh
```

The builder starts from the pinned CUDA 13.3 and PyTorch 2.13 base image,
checks all three integration locks, builds native extensions against the same
ABI, validates required launchers and imports, and runs source-composition
tests. Compiler metadata can produce a different Docker digest even when the
verified source trees match. Revision `26642da52049a41a25e425c630f94de113ea9e6a`
reproduces the immutable short-context image and does not include vLLM #418.

## Operational Limits

- The qualified topology is TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs.
- The immutable image is unsupported at prompt lengths of 63,744 tokens or
  more. vLLM #418 is required for long-context operation; 500,224 input tokens
  are model-qualified and 1,000,000 token positions are allocation-qualified.
- The scheduler permits one active sequence. Parallel request throughput has
  not been qualified for the vision-and-LMCache allocation.
- One image is permitted per request. Image preprocessing is bounded to
  40,960 input patches and 512 patches on one side.
- The qualified LMCache tier is CPU RAM only. The launcher implements a
  filesystem tier, but that tier is not part of this qualification.
- A host-local proxy exposes port 8000 on the qualification machine. The vLLM
  container binds port 8001; port 8000 is not created by the image.
- LLMConduit rewrites a `developer` role to `system` for the Kimi template and
  rejects unrecognized roles.
