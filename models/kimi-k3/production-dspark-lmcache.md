# Kimi-K3 MXFP4 Production Service

Status: **qualified** for TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs.

This service runs the official Kimi-K3 MXFP4 target with the Inferact DSpark
draft, native image input, structured tool calls, CPU LMCache, and
LLMConduit. The vLLM container exposes a physical 1,033,126-token KV cache and
a 1,000,000-token request limit.

The machine-readable qualification record is
[`validation/production-dspark-lmcache-clean-20260819.json`](validation/production-dspark-lmcache-clean-20260819.json).

For a RAM-only cache owned by vLLM instead of LMCache, use the separately
qualified [native host KV offload profile](native-host-kv-offload.md).

## Published artifacts

| Component | Immutable identity |
|---|---|
| vLLM image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-clean-vllm768a4fc-b12x4fd20fa-cu133-torch213-20260819-r3` |
| Docker digest | `sha256:5685a277bd5e51a433c51dfc45c248ddb340dacdc135682f74507015d2a0368d` |
| Docker recipe | `local-inference-lab/blackwell-llm-docker@e5dc1506ac3df4812a643f2ebc77c86399e88321` |
| vLLM tree | `768a4fc36104cb9ffa09d562192dd6deb5bbf3f2` |
| B12X tree | `4fd20fa4bf81c476d61af9dcd11d23cb6dc1ad5a` |
| LMCache tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| LLMConduit image | `voipmonitor/llmconduit@sha256:856b53ad893b47f7f868ac64ec899d3b23c89689e02cb85a108685e1eb05bc61` |
| Served model ID | `Kimi-K3-MXFP4-DSpark7-DCP16-1M` |

The image contains compiled installed packages. It does not activate a source
overlay and does not contain `/opt/kimi-k3-qsrt`. Package comparison against
the retained immutable Git checkouts produced identical manifests for 2,924
vLLM files, 237 B12X files, and 684 LMCache files. Compiled extension modules
are built from the same locked trees.

The runtime uses CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2,
FlashInfer 0.6.18, InstantTensor 0.1.9, and xgrammar 0.2.5.

## Runtime geometry

| Property | Value |
|---|---:|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Target expert weights | official MXFP4 |
| Draft checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| Tensor parallelism | 16 |
| Decode context parallelism | 16 |
| Maximum request length | 1,000,000 tokens |
| Physical KV capacity | 1,033,126 tokens |
| KV allocation | 1,325,000,000 bytes per GPU |
| KV dtype | FP8 |
| Scheduler batch-token limit | 1,024 |
| Active sequence limit | 1 |
| Images per prompt | 5 |
| LMCache L1 | 32 GiB host RAM |
| LMCache object size | 12,288 tokens |

The vision tower and multimodal projector are loaded on every rank. Their BF16
dense weights are converted to the runtime MXFP8 dense-linear representation;
routed language experts remain on the checkpoint's native MXFP4 path.

## Start vLLM, DSpark, vision, and LMCache

The Hugging Face cache must already contain the two pinned checkpoint
snapshots. Pulling by digest selects the qualified image exactly.

```bash
docker pull voipmonitor/vllm@sha256:5685a277bd5e51a433c51dfc45c248ddb340dacdc135682f74507015d2a0368d

mkdir -p /mnt/luke/kimi-k3-cache/kimi-k3-production-clean-768a4fc-4fd20fa

docker run -d \
  --name kimi-k3-production-dspark-lmcache-clean-768a4fc \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt label=disable \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/kimi-k3-production-clean-768a4fc-4fd20fa:/cache/jit:rw \
  voipmonitor/vllm@sha256:5685a277bd5e51a433c51dfc45c248ddb340dacdc135682f74507015d2a0368d
```

The image entrypoint is
`/usr/local/bin/serve-kimi-k3-production-dspark-ii`. It starts a CPU-only
LMCache server and then the 16 vLLM workers. Check readiness with:

```bash
docker logs -f kimi-k3-production-dspark-lmcache-clean-768a4fc
curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8100/healthcheck
```

The target checkpoint contains approximately 1.41 TiB of safetensors.
InstantTensor loaded it at approximately 9.5 GB/s; rank 0 completed model
loading in 183.64 seconds. CUDA Graph capture took 51 seconds and 0.29 GiB per
GPU.

### LMCache failure contract

LMCache is an optimization tier, not authoritative model state. A missing,
unavailable, or incomplete external-cache restore uses
`kv_load_failure_policy=recompute`; vLLM discards the affected external blocks
and computes them from the request tokens. A cache-tier failure must therefore
not become an empty API response.

The pickle transfer backend shares a Python/ZMQ control path with worker
heartbeats. TP16 restores can delay that control path without indicating a dead
worker. The production launcher uses a 30-second heartbeat interval and a
120-second worker reap timeout. These values prevent false worker eviction;
they do not make a failed request wait for the reap timeout.

The pickle backend in the published image is correctness-qualified but not a
low-latency restore implementation. It deserializes and dynamically pins
bounded CPU buffers. A pre-pinned tensor transport is required before treating
external restore latency as production-optimized.

## Start LLMConduit

Copy [`configs/llmconduit-production.yaml`](configs/llmconduit-production.yaml)
to `/root/vllm/kimi/llmconduit-kimi-k3-production.yaml`, then run:

```bash
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

LLMConduit preserves Kimi reasoning internally so the Kimi stream parser can
identify response and tool boundaries. Client reasoning controls affect what
is returned:

| Requested level | Backend level | Returned reasoning |
|---|---|---|
| `none` | thinking remains enabled internally | suppressed |
| `minimal`, `low`, `medium`, `high` | `high` | preserved |
| `xhigh`, `max` | `max` | preserved |
| omitted | `high` | preserved |

OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, streamed tool
arguments, and native image payloads are supported.

## Configure Oh My Pi

Status: **qualified** with OMP 17.3.5.

```bash
mkdir -p /root/.omp/agent
cp models/kimi-k3/configs/omp-models.yml /root/.omp/agent/models.yml
cp models/kimi-k3/configs/omp-config.yml /root/.omp/agent/config.yml

omp models --json
omp --thinking high
```

The global files under `/root/.omp/agent` are required. Project-local OMP
configuration applies only while OMP runs below that project directory.

OMP retains at most five images for this custom provider. The vLLM image limit
is also five, so replaying archived screenshots cannot exceed the server's
multimodal limit. LLMConduit turn capture replaces image bodies with
`<redacted uri>`; use
[`tools/replay-llmconduit-turn-capture.py`](tools/replay-llmconduit-turn-capture.py)
with one `--replacement-image-url` per redacted image when reproducing a turn.

## Qualification evidence

| Condition | Result |
|---|---|
| Package/source equality | vLLM 2,924/2,924, B12X 237/237, LMCache 684/684 files; all manifest hashes equal |
| Hybrid external-cache unit suite | 20 passed |
| Launcher configuration suite | passed |
| Container lifecycle | zero restarts; service remained healthy |
| GPU links under decode load | all 16 GPUs at PCIe Gen5 x16 and 100% utilization |
| Target-only normalized decode | 55.683 tok/s median; comparison 56.149 tok/s, difference -0.83% |
| DSpark normalized target throughput | 31.369 target cycles/s median over three 1,024-token runs |
| Comparison image `voipmonitor/vllm@sha256:a54da4e2432138d42334cac54555f3a51188489cb66029da0d96e4b39162d726` | 31.467 target cycles/s median; published compiled-package image difference -0.31% |
| DFlash normalized target throughput | 27.789 target cycles/s median; comparison 27.847 cycles/s, difference -0.21% |
| Coding decode | 124.34 active tok/s median over three 512-token runs |
| Unique short requests | 21/21 HTTP 200; no CUDA, cuBLAS, or engine errors |
| Five-image OMP replay | 134,219 prompt tokens, HTTP 200, `[DONE]`, no Kimi control-marker leak |
| LMCache repeat lookup | 122,880 aligned tokens found; 2,720 L1 chunks read |

Emitted DSpark throughput depends on the generated path because draft
acceptance changes with content. Target cycles per second are the primary
runtime regression metric. The three normalized clean-image runs emitted a
median 106.68 tok/s at 34.30% draft acceptance. A separate 2,048-token run
emitted 112.75 tok/s at 37.35% acceptance.

The 134,219-token repeat exercised the LMCache server lookup and read path, but
vLLM's local GPU prefix cache was also warm. It therefore does not isolate a
forced host-to-device restore. The 20-test external-cache suite independently
qualifies both synchronous and asynchronous `recompute` and `fail` policies,
invalid external blocks, and hybrid cache groups.

Host-local receipts are stored under:

```text
/mnt/luke/kimi-k3-runs/clean-r3-reference-ab-20260819
/mnt/luke/kimi-k3-runs/clean-r3-nospec-reference-20260819
/mnt/luke/kimi-k3-runs/clean-r3-dflash-reference-20260819
/mnt/luke/kimi-k3-runs/clean-r3-coding-reference-512-20260819
/mnt/luke/kimi-k3-runs/clean-r3-interleaved-query-cold-20260819
/mnt/luke/kimi-k3-runs/clean-r3-lmcache-recompute-qualification-20260819
/mnt/luke/kimi-k3-runs/clean-r3-gpu-link-load-20260819
```

## Comparison entrypoints

The same image contains these launchers:

| Entrypoint | Purpose |
|---|---|
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii` | official target without speculation; qualified at 55.683 tok/s with 1,460,937 physical KV tokens |
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii` | official target with Inferact DSpark; its production superset is qualified below |
| `/usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii` | official target with modal-labs DFlash; qualified at 90.626 emitted tok/s and 27.789 target cycles/s with 1,048,576 physical KV tokens |
| `/usr/local/bin/serve-kimi-k3-production-dspark-ii` | DSpark, vision, LMCache, and one-million-token allocation; qualified at 31.369 target cycles/s and 124.34 coding tok/s median |

Use a different writable `/cache/jit` directory for launchers with different
CUDA Graph shapes.

## Rebuild the image

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout e5dc1506ac3df4812a643f2ebc77c86399e88321

./build-kimi-k3-production-clean-runtime.sh
```

The builder validates the three source locks in
`patches/releases/kimi-k3-production-clean-build-20260819`, requires empty
source-patch lists, checks the resulting Git trees, compiles vLLM and B12X,
installs all Python packages into `/opt/venv`, and verifies imports without
`PYTHONPATH` source replacement. Build metadata can change the Docker digest;
the source tree labels remain the semantic build identity.

## Operational limits

- TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs is qualified.
- One million token positions are allocation-qualified. Coherent-document
  model quality above 500,224 tokens remains unsupported.
- The production scheduler permits one active sequence.
- A prompt may contain at most five images. Each encoded image is bounded to
  40,960 input patches and 512 patches on one side.
- The qualified LMCache tier is CPU RAM only. The filesystem tier is
  implemented but unsupported by this qualification.
- Port 8001 is the vLLM API, port 8100 is LMCache health/metrics, and port 8003
  is LLMConduit. Port 8000 is not created by this image.
