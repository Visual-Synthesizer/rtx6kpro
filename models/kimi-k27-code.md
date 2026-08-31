# Kimi-K2.7-Code on Black Benediction

Status: runbook only, no benchmark sweep yet. This page is derived from the
Kimi-K2.6 v9 Black Benediction runbook, but the target model is
`moonshotai/Kimi-K2.7-Code`.

This page intentionally starts with a target-only launch. Do not reuse the
Kimi-K2.6 DFlash draft from the v9 page with this target unless a matching
Kimi-K2.7-Code draft/speculator model is explicitly validated.

## Current Image

```text
voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612
```

DockerHub digest:

```text
sha256:0114603cf82584e3b14b91055243bb1daf59690a87500edbdee8592117ae93a4
```

## Component Revisions

| Component | Revision |
|---|---|
| Build recipe repo | `https://github.com/local-inference-lab/blackwell-llm-docker.git` |
| Build recipe commit | `f24137e95d09366e6ad8acf682cd87ce088bb903` |
| Build Dockerfile | `Dockerfile.vllm-b12x-cu132` |
| Build launcher | `build-black-benediction-b12xpr11-cu132.sh` |
| CUDA | `13.2.1` |
| cuBLAS active runtime | `13.4.1.2-1` |
| cuDNN apt package | `9.22.0.52-1` |
| PyTorch | `2.12.0+cu132` |
| Torch bundled NCCL | `2.29.7` |
| vLLM NCCL runtime | `/opt/libnccl-local-inference.so.2.30.4` |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical`, branch `canonical/cu132-nccl2304-amd-noxml` |
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/dflash-unified-bb-20260612` |
| vLLM commit | `72661cc997e8e775b982e2fd78b456092234da43` |
| vLLM package | `0.11.2.dev279+black.benediction.pr13.b12xw4a8.cu132.20260612` |
| B12X upstream branch | `dev/w4a8` |
| B12X upstream commit | `cce0a320b7b96cd8bd9f267ffb8c9e207af695be` |
| B12X overlay branch | `w4a8-mx-serving` |
| B12X overlay commit | `95dbdd13063d1d5d575c2cf1634e99370b81145b` |
| B12X package | `0.20.0` |
| FlashInfer branch | `refs/pull/3395/head` |
| FlashInfer commit | `b41aa8dd2fb93c49b1c6134bd1953040f8089d51` |
| FlashInfer package | `0.6.12+cu132` |
| DeepGEMM branch | `refs/pull/324/head` |
| DeepGEMM commit | `aced12c2c8882a945c568ace9d4a7e5778aae410` |

The overlay changes only the installed Python `b12x` package in the final image.
It adds the local `w4a8_mx` e8m0_k32 serving path and restores
`_COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS` to `256`.

## Build

Full image build:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout f24137e95d09366e6ad8acf682cd87ce088bb903

IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12xcce0a32-cu132-20260612 \
SYSTEM_BASE_IMAGE=voipmonitor/vllm:glm-kimi-cu132-system-base-20260608 \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:glm-kimi-cu132-build-base-20260608 \
BUILD_BASE_IMAGE=0 \
PUSH_BASE_IMAGE=0 \
PIN_SOURCE_COMMITS=1 \
MAX_JOBS=64 \
VLLM_MAX_JOBS=64 \
NVCC_THREADS=1 \
VLLM_NVCC_THREADS=1 \
VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
VLLM_REF=codex/dflash-unified-bb-20260612 \
VLLM_COMMIT=72661cc997e8e775b982e2fd78b456092234da43 \
LAUNCHER_REPO=https://github.com/local-inference-lab/vllm.git \
LAUNCHER_REF=codex/dflash-unified-bb-20260612 \
LAUNCHER_COMMIT=72661cc997e8e775b982e2fd78b456092234da43 \
B12X_REPO=https://github.com/lukealonso/b12x.git \
B12X_REF=dev/w4a8 \
B12X_COMMIT=cce0a320b7b96cd8bd9f267ffb8c9e207af695be \
VLLM_BUILD_VERSION=0.11.2.dev279+black.benediction.pr13.b12xw4a8.cu132.20260612 \
./build-black-benediction-b12xpr11-cu132.sh
```

Overlay build for the final image:

```bash
cd /root/vllm/worktrees/b12x-w4a8-cce0a32
git checkout w4a8-mx-serving
git rev-parse HEAD
# 95dbdd13063d1d5d575c2cf1634e99370b81145b

cat >/tmp/Dockerfile.kimi-b12x-overlay <<'EOF'
FROM voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12xcce0a32-cu132-20260612

COPY b12x /tmp/b12x-overlay/b12x

RUN python3 -c "import pathlib, shutil, b12x; dst = pathlib.Path(b12x.__file__).resolve().parent; shutil.rmtree(dst); shutil.copytree('/tmp/b12x-overlay/b12x', dst); print('overlaid b12x at', dst)" \
    && python3 -c "from b12x.attention.mla.compressed_config import _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS as rows; assert rows == 256, rows; print('b12x split max rows', rows)"

LABEL local-inference.b12x.branch="w4a8-mx-serving" \
      local-inference.b12x.upstream_commit="cce0a320b7b96cd8bd9f267ffb8c9e207af695be" \
      local-inference.b12x.overlay_commit="95dbdd13063d1d5d575c2cf1634e99370b81145b" \
      local-inference.b12x.overlay_note="Python overlay: w4a8_mx e8m0_k32 serving path plus MTP decode split rows un-revert"
EOF

docker build \
  -f /tmp/Dockerfile.kimi-b12x-overlay \
  -t voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612 \
  .

docker push voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612
```

Verify the final image:

```bash
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612

docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool

docker run --rm --entrypoint /bin/bash "$IMAGE" -lc '
python3 - <<PY
import importlib.metadata as md
from b12x.attention.mla.compressed_config import _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS
print("vllm", md.version("vllm"))
print("b12x", md.version("b12x"))
print("flashinfer", md.version("flashinfer-python"))
print("deep_gemm", md.version("deep_gemm"))
print("split_rows", _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS)
assert _COMPRESSED_MLA_DECODE_SPLIT_MAX_ROWS == 256
PY
'
```

## Target

| Item | Value |
|---|---|
| Target model | `moonshotai/Kimi-K2.7-Code` |
| Served name | `Kimi-K2.7-Code` |
| Port | `8402` |
| Default GPUs | `8,9,10,11,12,13,14,15` |
| TP | `8` |
| DCP | `1` |
| Runner | `VLLM_USE_V2_MODEL_RUNNER=1` |
| Target attention | `TRITON_MLA` |
| KV cache | `fp8` |
| PCIe custom allreduce | disabled, `VLLM_ENABLE_PCIE_ALLREDUCE=0` |
| Speculative decoding | disabled by default |

The image is self-contained. No vLLM or B12X bind mounts are required for this
profile.

## Docker Compose

Create the Compose file:

```bash
cat >/tmp/kimi-k27-code.compose.yaml <<'EOF'
services:
  kimi:
    image: ${IMAGE:-voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612}
    container_name: ${NAME:-kimi-k27-code}
    init: true
    network_mode: host
    ipc: host
    shm_size: 32g
    gpus: all
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 1048576
        hard: 1048576
      stack: 67108864
    volumes:
      - /root/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k27-code}:/cache
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUTE_DSL_ARCH: sm_120a
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_CACHE_DIR: /cache/jit/vllm
      TORCH_EXTENSIONS_DIR: /cache/torch_extensions
      TORCHINDUCTOR_CACHE_DIR: /cache/torchinductor
      TRITON_CACHE_DIR: /cache/triton
      XDG_CACHE_HOME: /cache
      FLASHINFER_WORKSPACE_BASE: /cache/jit/flashinfer
      NCCL_P2P_LEVEL: SYS
      NCCL_IB_DISABLE: "1"
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      SAFETENSORS_FAST_GPU: "1"
      TARGET: ${TARGET:-moonshotai/Kimi-K2.7-Code}
      PORT: ${PORT:-8402}
    command:
      - /bin/sh
      - -lc
      - |
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM
        exec vllm serve "$$TARGET" \
          --served-model-name Kimi-K2.7-Code \
          --host 0.0.0.0 \
          --port "$$PORT" \
          --trust-remote-code \
          --tensor-parallel-size 8 \
          --decode-context-parallel-size 1 \
          --kv-cache-dtype fp8 \
          --attention-backend TRITON_MLA \
          --gpu-memory-utilization 0.92 \
          --max-model-len 262144 \
          --max-num-seqs 64 \
          --max-num-batched-tokens 8192 \
          --max-cudagraph-capture-size 512 \
          --mm-processor-cache-gb 0 \
          --mm-encoder-tp-mode weights \
          --reasoning-parser kimi_k2 \
          --tool-call-parser kimi_k2 \
          --enable-auto-tool-choice \
          --async-scheduling \
          --enable-chunked-prefill \
          --enable-prefix-caching
EOF
```

Start:

```bash
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612 \
NAME=kimi-k27-code \
CACHE_ROOT=/root/.cache/vllm-kimi-k27-code \
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
PORT=8402 \
docker compose -f /tmp/kimi-k27-code.compose.yaml up -d
```

Readiness and KV cache:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k27-code 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Application startup complete' | tail -n 40
```

Stop:

```bash
docker compose -f /tmp/kimi-k27-code.compose.yaml down
```

## Command-Line Launch

```bash
#!/bin/bash
set -e

TARGET=moonshotai/Kimi-K2.7-Code
IMAGE=voipmonitor/vllm:black-benediction-pr13-b12xw4a8-vllm72661cc-b12x95dbdd1-cu132-overlay-20260612

for i in 1 2 3 4 5; do
  docker rm -f kimi-k27-code 2>/dev/null || true
  sleep 3
  if ! docker ps -a --format '{{.Names}}' | grep -q '^kimi-k27-code$'; then break; fi
done

docker run -d \
  --name kimi-k27-code \
  --init \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/vllm-kimi-k27-code:/cache \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e XDG_CACHE_HOME=/cache \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  "$IMAGE" \
  /bin/sh -lc 'unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_USE_B12X_FP8_GEMM; exec vllm serve "$@"' \
  -- \
  "$TARGET" \
  --served-model-name Kimi-K2.7-Code \
  --host 0.0.0.0 \
  --port 8402 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_MLA \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 512 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --async-scheduling \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

## Runtime Notes

- The first start on a fresh cache pays the full JIT cost. Reuse
  `/root/.cache/vllm-kimi-k27-code` for fast restarts.
- `NCCL_GRAPH_FILE` must be unset, not set to an empty path.
- `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` is intentionally unset for this profile.
- `VLLM_USE_B12X_FP8_GEMM` is intentionally unset for this profile.
- PCIe custom allreduce is disabled because it was slower on the Kimi V2
  DFlash profile than PYNCCL-only; keep it disabled for the first K2.7-Code
  baseline.
- Speculative decoding is intentionally disabled until a matching K2.7-Code
  draft/speculator model is selected and validated.

## Required Long-Context Fixes

Two vLLM fixes are required for correct DCP and long-context behaviour on
this image. Both are open against `dev/chthonic-consecration` and are NOT in
the pinned image; the sweep below ran with the fixed
`vllm/model_executor/layers/attention/mla_attention.py` bind-mounted over the
image.

| Fix | PR | Symptom without it |
|---|---|---|
| MLA DCP fp8 KV gather | [local-inference-lab/vllm#14](https://github.com/local-inference-lab/vllm/pull/14) | any DCP>1 chunked-context prefill crashes (`cp_gather_cache ... same dtype`) |
| MLA chunked-context merge strides | [local-inference-lab/vllm#15](https://github.com/local-inference-lab/vllm/pull/15) | every prompt above ~64k context tokens silently generates garbage (any DCP incl. 1; dense-MLA models only — sparse-MLA DS4 is unaffected) |

PR #15 also means dense-MLA quality/acceptance measurements taken on Black
Benediction images before 2026-06-12 with >64k-token contexts were generated
from corrupted prefill (throughput numbers remain valid).

## DCP Decode Sweep (2026-06-12)

Target-only (no speculative decoding), `GPU_MEM=0.94`, fresh container per
DCP value, PR #14 + #15 fixes mounted, otherwise the launch from this page.
Same methodology as the Kimi-K2.6 v8 sweep:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 --port 8402 --model Kimi-K2.7-Code \
  --concurrency 1,32 --contexts 0,128k --duration 30 \
  --skip-prefill --max-tokens 8192 --temperature 0 \
  --kv-budget <server KV tokens> --dcp-size <DCP> \
  --display-mode plain --no-hw-monitor \
  --output /root/bench-results/kimi-k27-dcp-sweep-20260612/dcp<DCP>/result.json
```

`∅` means the cell did not fit in the server KV cache and was skipped by the
benchmark (32 x 128k+8k > KV budget at every DCP).

| DCP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c32 tok/s | 128k/c1 tok/s | 128k/c32 tok/s | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.94 | 468,368 | 96.3 | 1,178.3 | 50.0 | ∅ | 0 |
| 2 | 0.94 | 936,864 | 85.4 | 1,035.8 | 59.3 | ∅ | 0 |
| 4 | 0.94 | 1,873,728 | 82.1 | 967.1 | 59.9 | ∅ | 0 |
| 8 | 0.94 | 3,747,456 | 86.3 | 835.5 | 69.9 | ∅ | 0 |

Trends match the K2.6 v8 sweep: DCP costs ~10-14% short-context single-stream
and batched throughput, and buys near-linear KV capacity plus +40% (DCP8)
single-stream decode at 128k.

Result JSONs:

```text
/root/bench-results/kimi-k27-dcp-sweep-20260612/
```

## 128k Coherence Validation (2026-06-12)

Per-DCP, on the sweep servers right after the decode run: a ~135k-token
prompt (cache-busted) with a needle on the first line, asking for the needle
plus a two-sentence description of the document; temp 0, 220 max tokens.
Checked: needle returned verbatim, unique-word ratio of the answer (degenerate
repetition shows up as <0.5), and CJK character count.

| DCP | prompt tokens | needle | uniq ratio | CJK | verdict |
|---:|---:|:---:|---:|---:|---|
| 1 | 134,789 | yes | 0.96 | 0 | coherent |
| 2 | 134,791 | yes | 0.96 | 0 | coherent |
| 4 | 134,790 | yes | 0.91 | 0 | coherent |
| 8 | 134,789 | yes | 0.92 | 0 | coherent |

All four answered in the same articulate form, e.g. DCP8:
"OCELOT-7291. This document mostly consists of the sentence 'The quick brown
fox jumps over the lazy dog' repeated many times. It appears to be a filler
or placeholder text with a single secret code at the beginning."

Without PR #15 the same probe fails for every prompt above ~65k tokens
(needle lost, degenerate repetition) at every DCP value including 1.

## DCP + K2.6 DFlash Draft Decode Sweep (2026-06-13)

Same sweep with the Kimi-K2.6 DFlash draft enabled
(`SubSir/Kimi-K2.6-DFlash-tmp`, `num_speculative_tokens=7`,
`attention_backend=TRITON_ATTN`), `GPU_MEM=0.90`. This required new vLLM
support for running DFlash under DCP (the draft KV cache is replicated on
every DCP rank instead of sharded; the draft's plain-attention backend
cannot reduce partial attention across ranks). All four DCP values boot,
pass the math smoke, and stay coherent at 128k.

| DCP | 0/c1 tok/s | 0/c32 tok/s | 128k/c1 tok/s | server accept rate | vs target-only 0/c1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 132.1 | 950.1 | 63.5 | 0.242 | +37% (96.3) |
| 2 | 91.6 | 660.9 | 29.8 | 0.081 | +7% (85.4) |
| 4 | 65.5 | 440.1 | 22.9 | 0.027 | −20% (82.1) |
| 8 | 54.1 | 275.9 | 29.5 | 0.038 | −37% (86.3) |

The K2.6 draft is **foreign** to the K2.7-Code target (no matching K2.7
DFlash export exists yet), so acceptance is low even at DCP1 (0.24) and
falls sharply as DCP grows. DFlash is a net win only at DCP1 with this
draft; at DCP≥4 it costs more than it saves. Generation stays correct at
every DCP (128k needle retrieval passes, no degeneration) — only the
speculative acceptance degrades. A native K2.7-Code draft would be needed
before recommending DFlash with DCP for this target.

128k coherence with DFlash (needle / unique-word ratio / CJK):

| DCP | needle | uniq ratio | CJK |
|---:|:---:|---:|---:|
| 1 | yes | 0.92 | 0 |
| 2 | yes | 0.88 | 0 |
| 4 | yes | 0.92 | 0 |
| 8 | yes | 0.92 | 0 |

Requires the DFlash-under-DCP vLLM change (`dcp_replicated` draft KV cache)
on top of PR #14 and PR #15. Result JSONs:

```text
/root/bench-results/kimi-k27-dcp-dflash-sweep-20260613/
```
