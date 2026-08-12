# Kimi-K3 TP16/DCP16 Runtime on RTX PRO 6000 Blackwell

Status: **qualified**.

This page specifies the containerized vLLM runtime for `moonshotai/Kimi-K3`
on 16 NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs. The target
checkpoint retains its MXFP4 expert weights and uses FP8 KV cache. One image
provides target-only decode, seven-token DSpark speculative decode, and
seven-token DFlash speculative decode.

## Qualified Artifact

| Item | Durable identifier |
|---|---|
| Docker image | `voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1` |
| Registry digest | `sha256:974edc237f27a4eaa83a53ce4927dd176a5ad8ce4fbb8d3d689fce82348531a5` |
| Local image ID used for qualification | `sha256:4be1d706e29cc5d53fc2891378ba185538d5a35e69793062a8f973f1886217f0` |
| Docker recipe | [`build/kimi-k3-ii-cu133-torch213-20260811@697f50f`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/697f50ff644f2c418645c64a50828dccce597d38) |
| Base image | `voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2` |
| Base image ID used for qualification | `sha256:651f0d37bc3da8469a2b9257bcff060f5509cc0d97c18f2bf340a6e5ce68d532` |
| Base image recipe | [`3775a0b`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/3775a0b6bcf30350c329f69ab55df7f3ecc8764b) |
| vLLM integration | [`integration/kimi-k3-ii-cu133-torch213-20260811@881ac39`](https://github.com/local-inference-lab/vllm/tree/881ac39a4fb6c5bbfa14f3944db560e0a27f3ffe) |
| vLLM review | [`local-inference-lab/vllm#284`](https://github.com/local-inference-lab/vllm/pull/284) |
| B12X review | [`local-inference-lab/b12x#147`](https://github.com/local-inference-lab/b12x/pull/147) |
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Runtime topology | TP16, DCP16, A2A DCP communication |
| Target KV dtype | FP8 |
| Weight loader | InstantTensor 0.1.9 |

The image uses CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, FlashInfer
0.6.15.post1, and B12X for target MLA decode.

## Source Composition

The image composes the following immutable source trees:

| Repository | Base | Applied changes | Resulting Git tree |
|---|---|---|---|
| `local-inference-lab/vllm` | `dev/infernal-invocation@c8d04a543e0e8b0896e60b8b11bec0bb2d780860` | Kimi-K3 TP16/DCP16 serving commit `881ac39a4fb6c5bbfa14f3944db560e0a27f3ffe` | `de04f08beb6ff0ef05189c31927f3a3320b1a6f1` |
| `local-inference-lab/b12x` | `master@184d7d52ad630841d0c6caf962f8b9d36f38992a` | PR [#124](https://github.com/local-inference-lab/b12x/pull/124), [#138](https://github.com/local-inference-lab/b12x/pull/138), and [#139](https://github.com/local-inference-lab/b12x/pull/139) | `2e6092a74d2449b8f8fa65d0c980533002db76cb` |

The B12X integration review [#147](https://github.com/local-inference-lab/b12x/pull/147)
has resulting tree `2e6092a74d2449b8f8fa65d0c980533002db76cb`, identical
to the B12X tree embedded in the qualified image.

The Docker recipe stores integration locks and verified patches under
`patches/releases/kimi-k3-infernal-invocation-runtime-r1/` and
`patches/releases/kimi-k3-hh-runtime-r1/`. The locks pin base commits, applied
PR heads, patch SHA-256 values, and resulting Git trees.

## Checkpoints and Runtime Weight Formats

Populate the Hugging Face cache mounted at `/root/.cache/huggingface` with the
target checkpoint and the draft checkpoint required by the selected profile:

| Role | Checkpoint | Runtime format |
|---|---|---|
| Target | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` | Checkpoint MXFP4 experts; InstantTensor load |
| DSpark draft | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` | Online MXFP8 linears; `fused_qkv_a_proj` remains BF16 |
| DFlash draft | `modal-labs/Kimi-K3-DFlash@c192d15a43407bf758b5ae0880d5c72052fef1de` | Online MXFP8 linears; `qkv_proj` remains BF16 |

The DSpark and DFlash launchers convert selected BF16 target KDA projections
to MXFP8 as each source tensor is loaded. They do not retain all BF16 KDA
projections simultaneously and do not move KDA projection repacking to a CPU
fallback. During conversion, GPU memory briefly contains the BF16 source,
MXFP8 output and scales, repack workspace, and InstantTensor staging buffers.

The DSpark cold-load qualification reached less than 21 MiB free memory per
GPU and emitted recoverable expandable-segment mapping warnings. Loading
completed, and the steady server passed decode qualification. One 4.45 GiB
DSpark draft tensor exceeds the 2.35 GiB InstantTensor ring and therefore uses
CPU safetensors; that tensor is not a target KDA projection.

## Cache Isolation and Port Selection

Each runtime profile requires an independent writable `/cache/jit` directory.
The profiles use different CUDA graph shapes and generated kernels; sharing a
JIT directory between profiles is unsupported.

```bash
mkdir -p /mnt/luke/kimi-k3-cache/infernal-release-20260812/nospec
mkdir -p /mnt/luke/kimi-k3-cache/infernal-release-20260812/dspark
mkdir -p /mnt/luke/kimi-k3-cache/infernal-release-20260812/dflash
```

The launch commands use host networking and set the vLLM server to port 8001.
The qualified host has a local proxy from host port 8000 to port 8001. On a
host where port 8000 is directly available, replace `-e PORT=8001` with
`-e PORT=8000`.

## Start Target-Only Decode

```bash
docker run -d \
  --name kimi-k3-infernal-nospec \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/infernal-release-20260812/nospec:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-nospec \
  voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1
```

The served model name is `Kimi-K3-MXFP4-DCP16-1M`.

## Start DSpark

The image defaults to the DSpark launcher. The explicit entrypoint below makes
the selected profile visible in container metadata.

```bash
docker run -d \
  --name kimi-k3-infernal-dspark \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/infernal-release-20260812/dspark:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-dspark \
  voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1
```

The served model name is `Kimi-K3-MXFP4-DSpark7-DCP16-1M`.

## Start DFlash

```bash
docker run -d \
  --name kimi-k3-infernal-dflash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/infernal-release-20260812/dflash:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-dflash \
  voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1
```

The served model name is `Kimi-K3-MXFP4-DFlash-DCP16-1M`.

## Readiness and Smoke Test

Wait for `Application startup complete` before sending a request:

```bash
docker logs -f kimi-k3-infernal-dspark
curl -fsS http://127.0.0.1:8000/v1/models | jq .
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K3-MXFP4-DSpark7-DCP16-1M",
    "prompt": "Write one sentence about speculative decoding.",
    "max_tokens": 32,
    "temperature": 0
  }'
```

## Qualified Capacity and Decode Throughput

Measurements used 16 RTX PRO 6000 Blackwell Workstation Edition GPUs, TP16/DCP16,
one active request, a 256-token prompt, and 1,024 generated tokens. The
target-only protocol constrained generation to token ID 13 so model output did
not affect per-token timing. Speculative measurements used greedy sampling.

| Profile | Reported target KV tokens | Decode tok/s median | Target cycles/s median | Draft acceptance median | Measured requests |
|---|---:|---:|---:|---:|---:|
| Target-only | 1,460,937 | 56.05 | n/a | n/a | 8 |
| DSpark K7 | 1,057,049 | 118.92 | 30.90 | 0.409 | 8 |
| DFlash K7 | 1,048,576 | 89.35 | 30.34 | 0.277 | 6 |

The DSpark model emitted a median 3.865 tokens per target cycle. The DFlash
model emitted a median 2.940 tokens per target cycle. Target weight loading
took 161.42 seconds for target-only, 165.29 seconds for DSpark, and 164.01
seconds for DFlash. Complete model loading took 185.96, 186.99, and 190.56
seconds, respectively.

## Sieve Coding Acceptance

Status: **qualified** for DSpark K7 and DFlash K7.

The Sieve coding acceptance protocol sends one streamed chat request with the
exact user prompt:

```text
Write a Python script that implements the Sieve of Eratosthenes.
```

Each profile received one discarded warm-up request followed by three measured
requests. Every request used one active sequence, no synthetic input context,
and `max_tokens=2000`. The request omitted both `temperature` and `seed`.
Reasoning and final-content deltas were both included in the output integrity
scan. The harness computed generation rate from the server-reported completion
token count divided by elapsed time from the first streamed token through
stream completion.

The qualification harness was `llm_cjk_watchdog 0.1.0` at `/mnt/test.py`, with
SHA-256
`5246883eaaada9bd93a361eaffcfb0ee0e2c3a331230c523967f3c36acabdc11`.
Its request protocol and throughput calculation match
[`llm_cjk_watchdog.py@86cf05c`](https://github.com/local-inference-lab/llm-inference-bench/blob/86cf05c2f42f4d21b909b6e684424ca1aab89fd5/llm_cjk_watchdog.py);
the qualification copy also records reasoning, final content, finish reason,
usage, and timing in JSON.

| Profile | Run | Completion tokens | Generation tok/s | TTFT | Finish reason | Han ideographs | Python validation |
|---|---:|---:|---:|---:|---|---:|---|
| DSpark K7 | 1 | 2,000 | 124.72 | 0.292 s | `length` | 0 | pass |
| DSpark K7 | 2 | 1,885 | 134.26 | 0.293 s | `stop` | 0 | pass |
| DSpark K7 | 3 | 1,612 | 138.73 | 0.294 s | `stop` | 0 | pass |
| DFlash K7 | 1 | 1,190 | 146.95 | 0.301 s | `stop` | 0 | pass |
| DFlash K7 | 2 | 1,116 | 150.09 | 0.301 s | `stop` | 0 | pass |
| DFlash K7 | 3 | 1,893 | 127.56 | 0.300 s | `stop` | 0 | pass |

Median generation rate was **134.26 tok/s for DSpark K7** and **146.95 tok/s
for DFlash K7**. All six final answers contained a syntactically valid first
fenced Python block. Executing each block and asserting
`sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]` passed.
No CJK Unified Ideographs, extensions A-E, or Compatibility Ideographs appeared
in reasoning or final content. DSpark run 1 completed its Python program before
the 2,000-token limit truncated the explanatory prose.

Sieve generation rate is output-dependent because speculative acceptance and
completion length vary with generated tokens. It does not replace the
fixed-token normalized throughput table above. In particular, DFlash was
faster by median generation rate across its three Sieve responses, while
DSpark was faster under the normalized 1,024-token protocol.

The machine-readable per-run results and evidence hashes are stored in
[`validation/sieve-speculative-decode-20260812.json`](validation/sieve-speculative-decode-20260812.json).
Raw JSON summaries and streamed output are stored on the qualification host
under:

```text
/mnt/luke/kimi-k3-runs/infernal-release-20260812/dspark/sieve
/mnt/luke/kimi-k3-runs/infernal-release-20260812/dflash/sieve
```

To reproduce the request protocol with the repository harness, set `MODEL` to
the served name for the selected profile and run the command three times after
one unrecorded warm-up:

```bash
git clone https://github.com/local-inference-lab/llm-inference-bench.git
git -C llm-inference-bench checkout 86cf05c2f42f4d21b909b6e684424ca1aab89fd5

python3 llm-inference-bench/llm_cjk_watchdog.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model "$MODEL" \
  --max-tokens 2000 \
  --no-overlay
```

## Build the Image

Pull the byte-identical qualified runtime by digest:

```bash
docker pull \
  voipmonitor/vllm@sha256:974edc237f27a4eaa83a53ce4927dd176a5ad8ce4fbb8d3d689fce82348531a5
```

The following commands rebuild the pinned base and runtime source
compositions. Native compiler metadata can make the rebuilt image digest
differ even when all pinned source trees and runtime interfaces match.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

git checkout 3775a0b6bcf30350c329f69ab55df7f3ecc8764b
IMAGE=voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2 \
RELEASE_DATE=20260811 \
REVISION=r2 \
./build-kimi-k3-cu133-torch213-base.sh

git checkout 697f50ff644f2c418645c64a50828dccce597d38
BASE_IMAGE=voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2 \
IMAGE=voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1 \
RELEASE_DATE=20260812 \
REVISION=r1 \
./build-kimi-k3-infernal-invocation-cu133-torch213.sh
```

The base builder starts from
`nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`.
It builds PyTorch 2.13.0 from
`cf30153c4c131c8164ee7798e5022d810682e2cb`, Torchvision 0.28.0 from
`8fb87713a24951e639c494b0f2a8a81b5f8e33a6`, patched NCCL 2.31.2 from
`fb6f40999a2a9e63104d4ae4a84118bce61528f8`, and XGrammar 0.2.5 from
`2ea71da4ccb997a06928c9fb69b99f330da56697`.

The base builder runs CUDA, Torchvision, and 16-rank NCCL smoke tests. The
runtime builder verifies the vLLM and B12X integration patch hashes and Git
trees, builds the native extensions, verifies installed package versions and
imports, and repeats the 16-rank NCCL smoke test. Set `PUSH_IMAGE=1` on either
builder to publish its output after validation.

## Evidence

The machine-readable qualification receipt is
[`validation/kimi-k3-infernal-invocation-runtime-20260812.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/7ae7b4ef40ed0b7c3c4cd6551ecb3dfc81578f3e/validation/kimi-k3-infernal-invocation-runtime-20260812.json).
Server logs, container inspections, and normalized benchmark summaries from
the qualified host are stored under:

```text
/mnt/luke/kimi-k3-runs/infernal-release-20260812
```

The host-local evidence is not required to rebuild the image. The integration
locks and qualification receipt contain the durable source and result
identities.
