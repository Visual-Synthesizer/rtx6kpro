# Kimi-K2.6 v5 on 8x RTX PRO 6000 Blackwell

Status: in progress, created 2026-05-20. This page is the current canonical
Kimi-K2.6 v5 recipe for the CUDA 13.2 upstream-rebased vLLM stack on the local
8x RTX PRO 6000 Blackwell PCIe host.

The runtime recipe below is intentionally explicit. It is the configuration we
use for the v5 measurements: vLLM V2 model runner, TRITON_MLA, fp8 target KV,
fp8 Eagle3 draft KV, standard+greedy verifier-backed MTP, FlashInfer autotune,
and PCIe custom allreduce disabled.

## Image

```bash
voipmonitor/vllm:kimi-v5-cu132-89da7631
```

Image metadata:

| Component | Revision |
|---|---|
| CUDA base | `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` |
| CUDA | `13.2.1` |
| cuDNN | `9.20.0.48` |
| PyTorch | `2.12.0+cu132` |
| NCCL | `local-inference-lab/nccl-canonical`, branch `canonical/cu132-nccl2304-amd-noxml`, version `2.30.4` |
| vLLM repo | `https://github.com/voipmonitor/vllm.git` |
| vLLM branch | `codex/glm-kimi-upstream-rebase-20260519` |
| vLLM commit | `89da7631ebb844d39dcd5abe5265bc20983be69f` |
| local-inference branch mirror | `https://github.com/local-inference-lab/vllm/tree/dev/kimi-v5-cu132-upstream-rebase` |
| FlashInfer | `flashinfer-ai/flashinfer`, branch `main`, commit `9035311e975a6aeb2d229f5162e999dfb7c9a733` |
| B12X | `local-inference-lab/b12x`, branch `codex/glm51-kimi-b12x-a16-cpuhangfix-cutedsl45-20260512`, commit `c929144c7689668b07ca65af10ceadf1c745165d` |

Build command used for this image family:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout codex/glm-kimi-cu132-clean
git pull --ff-only

# Optional exact Dockerfile repo pin used for this page:
git checkout 4f5a95384446ef9c1b966a456cc12fff5db0999b

IMAGE=voipmonitor/vllm:kimi-v5-cu132-89da7631

IMAGE="$IMAGE" \
MAX_JOBS=128 VLLM_MAX_JOBS=128 NVCC_THREADS=1 VLLM_NVCC_THREADS=1 \
VLLM_REF=codex/glm-kimi-upstream-rebase-20260519 \
VLLM_COMMIT=89da7631ebb844d39dcd5abe5265bc20983be69f \
LAUNCHER_REF=codex/glm-kimi-upstream-rebase-20260519 \
LAUNCHER_COMMIT=89da7631ebb844d39dcd5abe5265bc20983be69f \
FLASHINFER_REF=main \
FLASHINFER_COMMIT=9035311e975a6aeb2d229f5162e999dfb7c9a733 \
B12X_REF=codex/glm51-kimi-b12x-a16-cpuhangfix-cutedsl45-20260512 \
B12X_COMMIT=c929144c7689668b07ca65af10ceadf1c745165d \
VLLM_BUILD_VERSION=0.20.2+glmkimi.v5.cu132.89da7631.20260520 \
./build-glm-kimi-cu132.sh
```

Verify after build:

```bash
docker run --rm "$IMAGE" /usr/local/bin/verify-glm-kimi-cu132
docker run --rm "$IMAGE" python -c "import importlib.metadata as md; print(md.version('humming-kernels')); print(md.version('b12x'))"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
```

## What Changed Versus v4

| Area | v4 | v5 |
|---|---|---|
| CUDA stack | CUDA 13.0/13.1-era image family | CUDA 13.2.1, cuDNN 9.20, PyTorch 2.12.0+cu132 |
| vLLM base | Older GLM/Kimi rebase branch | Rebased onto newer upstream vLLM, commit `89da7631` image |
| NCCL | Patched PR2127-style no-XML NCCL 2.30.3 | Canonical no-XML NCCL 2.30.4, `libnccl-local-inference.so.2.30.4` |
| model runner | Previous runner path | V2 model runner enabled with `VLLM_USE_V2_MODEL_RUNNER=1` |
| FlashInfer | Earlier git revision and partial autotune coverage | FlashInfer main plus vLLM warmup/autotune bucket patches |
| MTP correctness | Production-style fast speculative path used verifier-backed speculative decoding | Both greedy/strict and probabilistic draft sampling are lossless in vLLM; v5 default should be chosen by measured acceptance and throughput |
| allreduce | v4 production generally used vLLM C++ PCIe custom allreduce | v5 default is AR off because the measured DCP8 and mixed DCP1 profile is faster or safer with NCCL fallback |
| reasoning/tool parser | Optional or image dependent | V2 now supports serving-layer `--reasoning-parser kimi_k2`; tool calls require `--tool-call-parser kimi_k2 --enable-auto-tool-choice` |

## MTP Policy

The measured v5 speculative path is verifier-backed and lossless in vLLM. There
are two relevant draft-sampling modes:

| Mode | Config | Meaning | Tradeoff |
|---|---|---|---|
| Greedy/strict | `rejection_sample_method=standard`, `draft_sample_method=greedy` | Draft proposes argmax tokens. The verifier treats the draft distribution as one-hot: accept with target probability of that token, otherwise recover from target distribution with the draft token removed. | Lowest overhead, often fastest. Acceptance depends on how often high-probability target tokens match the draft argmax. |
| Probabilistic/full-q | `rejection_sample_method=standard`, `draft_sample_method=probabilistic` | Draft samples from its own distribution and caches draft logits/probs so the verifier can use the full `p/q` probability ratio and residual `max(p-q,0)`. | Usually improves acceptance for stochastic draft sampling, but costs extra memory and work for draft logits/probs. |

The current runbook uses standard+greedy. On the DCP8 `cc1/ctx0` 5-run A/B,
standard+greedy averaged `102.7 tok/s` with `0.339` acceptance, while
probabilistic/full-q averaged `98.3 tok/s` with `0.331` acceptance. That makes
standard+greedy the default until a workload shows probabilistic/full-q winning
on both acceptance and throughput.

```json
{
  "model": "festr2/kimi-k2.6-eagle3-mla-fp8",
  "method": "eagle3",
  "num_speculative_tokens": 3,
  "draft_attention_backend": "TRITON_MLA",
  "draft_kv_cache_dtype": "fp8",
  "rejection_sample_method": "standard",
  "draft_sample_method": "greedy"
}
```

In this context:

| Symbol | Meaning |
|---|---|
| `p` | Target Kimi distribution for the next token after applying the request sampling params, for example temperature 1. |
| `q` | Draft/Eagle distribution for the proposed token after applying the same sampling params. |

For probabilistic/full-q draft sampling, the verifier accepts a proposed draft
token with probability `min(1, p/q)`. If a token is rejected, the replacement
token is sampled from the residual distribution proportional to `max(p - q, 0)`.

For greedy/strict draft sampling, `q` is one-hot for the proposed draft token.
That means the accept probability is simply the target probability of the
proposed draft token, and the recovery distribution is the target distribution
with that proposed token removed. This is still lossless; it is not a
"wrong-distribution" mode.

Practical meaning:

- With `temperature=0`, greedy target sampling dominates and both modes should
  produce the same target greedy output up to normal numeric/system effects.
- With `temperature=1` or other stochastic sampling, both modes are intended to
  preserve the target model output distribution in vLLM. They can produce
  different sampled token streams because sampling is stochastic, but neither
  mode should bias the final distribution if the verifier/recovery path is
  functioning correctly.
- If greedy/strict is faster and has equal or better acceptance on Kimi, use it.
  The reason to keep probabilistic/full-q is empirical acceptance/throughput, not
  correctness.

The important runtime detail is that the image launcher has a simpler default
MTP config. The v5 V2 recipe below bypasses the launcher, calls
`/opt/venv/bin/vllm serve` directly, enables the Kimi reasoning and tool-call
parsers explicitly, and passes the measured `KIMI_SPEC_CONFIG`.

## Allreduce Policy

Current v5 default: disable PCIe custom allreduce.

```bash
VLLM_ENABLE_PCIE_ALLREDUCE=0
```

Reason: the v5 AR screen showed AR-off is the best DCP8 setting for the target
V2 + MTP profile. DCP1 has one narrow long-context case where the
old v4 C++ AR policy can win slightly, but its short-context loss is larger.
The default therefore stays AR-off until a later workload-specific AR policy is
proven across the full matrix.

Do not inherit the image default here: the image default still enables C++ AR
for compatibility with older recipes. The v5 Kimi run command below overrides
it to off.

## Runtime Profiles

Use these profile values unless doing a strict A/B:

| Profile | `DCP` | `MTP` | `GPU_MEM` | Notes |
|---|---:|---:|---:|---|
| DCP1 + MTP | 1 | 1 | 0.90 | v5 standard+greedy speculative profile |
| DCP1 no-MTP | 1 | 0 | 0.94 | target-only baseline, larger KV |
| DCP2 + MTP | 2 | 1 | 0.90 | v5 standard+greedy speculative profile |
| DCP2 no-MTP | 2 | 0 | 0.94 | target-only baseline, larger KV |
| DCP4 + MTP | 4 | 1 | 0.90 | v5 standard+greedy speculative profile |
| DCP4 no-MTP | 4 | 0 | 0.94 | target-only baseline, larger KV |
| DCP8 + MTP | 8 | 1 | 0.90 | primary v5 standard+greedy long-context profile |
| DCP8 no-MTP | 8 | 0 | 0.94 | target-only baseline, larger KV |

If you need an exact MTP on/off A/B at identical memory pressure, set
`GPU_MEM=0.90` for both.

## Docker Run

Create the launcher:

```bash
cat >/tmp/run-kimi-k26-v5 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:kimi-v5-cu132-89da7631}"
NAME="${NAME:-kimi-k26-v5}"
PORT="${PORT:-8402}"
DCP="${DCP:-8}"
MTP="${MTP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
CACHE_ROOT="${CACHE_ROOT:-${HOME}/.cache/vllm-kimi-k26-v5}"
MODEL_PATH="${MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}"

SPEC_CONFIG='{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'

mkdir -p \
  "${CACHE_ROOT}/cutlass_dsl" \
  "${CACHE_ROOT}/jit" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/vllm"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

mtp_disable=0
if [[ "${MTP}" == "0" ]]; then
  mtp_disable=1
fi

exec docker run -d --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ipc=host --network host --privileged \
  --name "${NAME}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${CACHE_ROOT}/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "${CACHE_ROOT}/jit:/cache/jit" \
  -v "${CACHE_ROOT}/triton:/root/.cache/triton" \
  -v "${CACHE_ROOT}/torchinductor:/root/.cache/torchinductor" \
  -v "${CACHE_ROOT}/vllm:/root/.cache/vllm" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e USE_NCCL_XML=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e PORT="${PORT}" \
  -e MODEL="${MODEL_PATH}" \
  -e SERVED_MODEL_NAME=Kimi-K2.6 \
  -e TP_SIZE=8 \
  -e DCP_SIZE="${DCP}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEM}" \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e MAX_NUM_SEQS=128 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
  -e ATTENTION_BACKEND=TRITON_MLA \
  -e KV_CACHE_DTYPE=fp8 \
  -e LOAD_FORMAT=fastsafetensors \
  -e ENABLE_PREFIX_CACHING=1 \
  -e ENABLE_CHUNKED_PREFILL=1 \
  -e ENABLE_ASYNC_SCHEDULING=1 \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc "$(cat <<RUN_EOF
set -euo pipefail
spec_args=()
if [[ "${mtp_disable}" != "1" ]]; then
  spec_args+=(--speculative-config '${SPEC_CONFIG}')
fi
exec /opt/venv/bin/vllm serve '${MODEL_PATH}' \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port '${PORT}' \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size '${DCP}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization '${GPU_MEM}' \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --enable-flashinfer-autotune \
  --max-cudagraph-capture-size 512 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  "\${spec_args[@]}"
RUN_EOF
)"
EOF
chmod +x /tmp/run-kimi-k26-v5
```

Examples:

```bash
# Primary v5 profile: DCP8, standard+greedy MTP on, AR off.
PORT=8402 DCP=8 MTP=1 GPU_MEM=0.90 /tmp/run-kimi-k26-v5

# DCP4 target-only baseline.
PORT=8402 DCP=4 MTP=0 GPU_MEM=0.94 /tmp/run-kimi-k26-v5
```

Readiness:

```bash
curl -fsS http://127.0.0.1:8402/v1/models | jq .
docker logs kimi-k26-v5 2>&1 | rg 'Application startup complete|GPU KV cache size|speculative_config|Custom allreduce'
```

Expected checks:

```text
vLLM version includes 0.20.2+glmkimi.v5.cu132.89da7631.20260520
decode_context_parallel_size=<DCP>
speculative_config includes festr2/kimi-k2.6-eagle3-mla-fp8 when MTP=1
Custom allreduce is disabled when VLLM_ENABLE_PCIE_ALLREDUCE=0
structured_outputs_config includes reasoning_parser='kimi_k2'
tool_call_parser is kimi_k2 and auto tool choice is enabled
Application startup complete.
```

Reasoning and tool-call smoke test:

```bash
curl -fsS http://127.0.0.1:8402/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K2.6",
    "messages": [
      {
        "role": "user",
        "content": "Use the weather tool to get the current weather in Prague. Do not answer in text; call the tool."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256
  }' | jq '.choices[0].finish_reason, .choices[0].message.reasoning, .choices[0].message.tool_calls'
```

Expected result: `finish_reason` is `tool_calls`, `message.reasoning` is
populated, `message.content` is `null`, and the tool call arguments are clean
JSON, for example `{"city":"Prague"}`.

## Docker Compose

The same profile can be launched with Docker Compose. Runtime knobs are still
environment variables so the same compose file can run DCP1/2/4/8 and MTP on/off.

```bash
cat >/tmp/kimi-k26-v5.compose.yaml <<'EOF'
services:
  kimi-k26-v5:
    image: ${IMAGE:-voipmonitor/vllm:kimi-v5-cu132-89da7631}
    container_name: ${NAME:-kimi-k26-v5}
    network_mode: host
    ipc: host
    privileged: true
    gpus: all
    entrypoint: ["/bin/bash", "-lc"]
    command: |
      set -euo pipefail
      spec_args=()
      if [[ "$${KIMI_DISABLE_MTP:-0}" != "1" ]]; then
        spec_args+=(--speculative-config "$${KIMI_SPEC_CONFIG}")
      fi
      exec /opt/venv/bin/vllm serve "$${MODEL}" \
        --served-model-name Kimi-K2.6 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$${PORT}" \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 1 \
        --decode-context-parallel-size "$${DCP_SIZE}" \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --async-scheduling \
        --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
        --max-model-len 262144 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 128 \
        --mm-processor-cache-gb 0 \
        --mm-encoder-tp-mode weights \
        --attention-backend TRITON_MLA \
        --kv-cache-dtype fp8 \
        --enable-flashinfer-autotune \
        --max-cudagraph-capture-size 512 \
        --reasoning-parser kimi_k2 \
        --tool-call-parser kimi_k2 \
        --enable-auto-tool-choice \
        "$${spec_args[@]}"
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v5}/cutlass_dsl:/root/.cache/cutlass_dsl
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v5}/jit:/cache/jit
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v5}/triton:/root/.cache/triton
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v5}/torchinductor:/root/.cache/torchinductor
      - ${CACHE_ROOT:-/root/.cache/vllm-kimi-k26-v5}/vllm:/root/.cache/vllm
    environment:
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      NVIDIA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
      VLLM_USE_V2_MODEL_RUNNER: "1"
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      USE_NCCL_XML: "0"
      VLLM_ENABLE_PCIE_ALLREDUCE: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD: "0"
      VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER: "0"
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
      VLLM_DISABLED_KERNELS: MarlinFP8ScaledMMLinearKernel
      PORT: ${PORT:-8402}
      MODEL: ${MODEL_PATH:-/root/.cache/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/b5aabbfb20227ed42becbf5541dbffd213942c58}
      SERVED_MODEL_NAME: Kimi-K2.6
      TP_SIZE: "8"
      DCP_SIZE: ${DCP:-8}
      GPU_MEMORY_UTILIZATION: ${GPU_MEM:-0.90}
      MAX_MODEL_LEN: "262144"
      MAX_NUM_BATCHED_TOKENS: "8192"
      MAX_NUM_SEQS: "128"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "512"
      ATTENTION_BACKEND: TRITON_MLA
      KV_CACHE_DTYPE: fp8
      LOAD_FORMAT: fastsafetensors
      ENABLE_PREFIX_CACHING: "1"
      ENABLE_CHUNKED_PREFILL: "1"
      ENABLE_ASYNC_SCHEDULING: "1"
      KIMI_DISABLE_MTP: ${KIMI_DISABLE_MTP:-0}
      KIMI_ENABLE_FLASHINFER_AUTOTUNE: "1"
      KIMI_SPEC_CONFIG: '{"model":"festr2/kimi-k2.6-eagle3-mla-fp8","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"standard","draft_sample_method":"greedy"}'
EOF
```

Run:

```bash
# DCP8 + MTP on
DCP=8 KIMI_DISABLE_MTP=0 GPU_MEM=0.90 docker compose -f /tmp/kimi-k26-v5.compose.yaml up -d

# DCP1 no-MTP
DCP=1 KIMI_DISABLE_MTP=1 GPU_MEM=0.94 docker compose -f /tmp/kimi-k26-v5.compose.yaml up -d
```

Stop:

```bash
docker compose -f /tmp/kimi-k26-v5.compose.yaml down
```

## Benchmark Method

Use `/root/llm-inference-bench/llm_decode_bench.py` v0.4.20 or newer. For vLLM,
let the benchmark infer KV budget from `/metrics`; always pass the correct
`--dcp-size`.

Current limited v5 sweep:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,32 \
  --contexts 0,128k \
  --duration 30 \
  --skip-prefill \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v5-standard-greedy-matrix-89da7631-20260520/dcp<DCP>/mtp<MTP>/result.json
```

Full matrix to run later:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 8402 \
  --model Kimi-K2.6 \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --skip-prefill \
  --dcp-size <DCP> \
  --display-mode plain \
  --no-hw-monitor \
  --output /root/bench-results/kimi-v5-full-matrix-89da7631-20260520/dcp<DCP>/mtp<MTP>/result.json
```

## Standard+Greedy Limited Sweep Results

Result directory:

```bash
/root/bench-results/kimi-v5-standard-greedy-matrix-89da7631-20260520
```

These are the current v5 publishable numbers. They were collected on 2026-05-20
with 30s sustained decode cells, AR off, V2 model runner, `ctx0` and `ctx128k`,
`cc1` and `cc32`, and standard+greedy MTP for `MTP=1`.

```bash
IMAGE=voipmonitor/vllm:kimi-v5-cu132-89da7631 \
DURATION=30 \
DCP_LIST="1 2 4 8" \
MTP_LIST="0 1" \
/root/run_kimi_v5_standard_greedy_matrix.sh
```

`acc` is the average speculative acceptance rate reported by server metrics for
that measured cell. For `MTP=0`, acceptance is `0.000` by definition. `N/A`
means the cell was skipped because it did not fit in KV cache. For `128k/c32`,
the request would require about 4.26M KV tokens, above the measured capacity of
all profiles in this limited sweep.

### DCP 1

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 491,344 | 89.7 | 0.000 | 999.4 | 0.000 | 45.5 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 342,672 | 139.4 | 0.385 | 1166.5 | 0.365 | 58.3 | 0.370 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 2

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 982,688 | 79.2 | 0.000 | 885.4 | 0.000 | 54.5 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 685,344 | 122.4 | 0.415 | 1050.2 | 0.356 | 71.8 | 0.390 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 4

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 1,965,376 | 76.1 | 0.000 | 835.4 | 0.000 | 54.8 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 1,370,688 | 105.8 | 0.265 | 966.3 | 0.391 | 62.0 | 0.417 | N/A | N/A | AR off, standard+greedy MTP |

### DCP 8

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 3,930,752 | 74.8 | 0.000 | 688.4 | 0.000 | 60.8 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 2,741,376 | 108.6 | 0.347 | 697.8 | 0.426 | 71.1 | 0.363 | N/A | N/A | AR off, standard+greedy MTP |

## Appendix: Previous Probabilistic/full-q Limited Sweep

Result directory:

```bash
/root/bench-results/kimi-v5-limited-matrix-42caa6d38-20260520
```

These historical numbers were collected on the previous v5 image
`voipmonitor/vllm:glm-kimi-upstream-rebase-cu132-vllm42caa6d38-20260520`.
The current recommended runtime image is
`voipmonitor/vllm:kimi-v5-cu132-89da7631`; the main tables above supersede this
appendix once the standard+greedy sweep completes.

The DCP8/MTP=1 row was remeasured once after the initial sweep because the
first `0/c1` value looked low. That rerun is stored in:

```bash
/root/bench-results/kimi-v5-rerun-dcp8-mtp1-aroff-20260520
```

`acc` is the average speculative acceptance rate reported by server metrics for
that measured cell. For `MTP=0`, acceptance is `0.000` by definition. `N/A`
means the cell was skipped because it did not fit in KV cache. For `128k/c32`,
the request would require about 4.26M KV tokens, above the measured capacity of
all profiles in this limited sweep.

### DCP 1

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 491,344 | 89.7 | 0.000 | 993.8 | 0.000 | 45.6 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 320,816 | 134.5 | 0.443 | 1093.2 | 0.323 | 55.5 | 0.247 | N/A | N/A | AR off, probabilistic/full-q MTP |

### DCP 2

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 982,688 | 79.2 | 0.000 | 881.0 | 0.000 | 54.4 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 685,344 | 115.5 | 0.392 | 994.7 | 0.306 | 68.2 | 0.314 | N/A | N/A | AR off, probabilistic/full-q MTP |

### DCP 4

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 1,965,376 | 76.1 | 0.000 | 831.3 | 0.000 | 54.8 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 1,370,688 | 111.2 | 0.321 | 906.5 | 0.359 | 59.7 | 0.287 | N/A | N/A | AR off, probabilistic/full-q MTP |

### DCP 8

| MTP | GPU mem | KV cache tokens | 0/c1 tok/s | 0/c1 acc | 0/c32 tok/s | 0/c32 acc | 128k/c1 tok/s | 128k/c1 acc | 128k/c32 tok/s | 128k/c32 acc | Notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.94 | 3,930,752 | 74.9 | 0.000 | 693.2 | 0.000 | 60.9 | 0.000 | N/A | N/A | AR off, no MTP |
| 1 | 0.90 | 2,566,528 | 98.4 | 0.333 | 623.8 | 0.346 | 52.6 | 0.525 | N/A | N/A | AR off, probabilistic/full-q MTP, rerun |

## Notes And Risks

- vLLM documents speculative decoding as lossless up to normal hardware
  numerical limits. For Kimi v5, choose greedy/strict vs probabilistic/full-q by
  benchmarked acceptance and throughput, not by output-quality assumptions.
- `VLLM_ENABLE_PCIE_ALLREDUCE=0` must be set explicitly. The image default still
  enables the older C++ AR path.
- Persistent cache mounts matter. Keep `/cache/jit`, Triton, TorchInductor,
  CUTE DSL, and vLLM cache directories mounted across restarts to avoid repeated
  compile/autotune cost.
- This V2 runbook intentionally enables `--reasoning-parser kimi_k2`,
  `--tool-call-parser kimi_k2`, and `--enable-auto-tool-choice`. Reasoning
  parsing and Kimi tool calls are supported in image `kimi-v5-cu132-89da7631`;
  request-level `thinking_token_budget` remains unsupported in V2.
- Tool calls are validated with `tool_choice="auto"`. Forced named
  `tool_choice` is not the recommended Kimi smoke-test path because the current
  vLLM serving path can wrap the raw Kimi tool-call marker body into
  `arguments`.
- For acceptance-rate work, use the same draft model and the same
  `KIMI_SPEC_CONFIG`. The launcher default draft is not the v5 measured fp8
  draft.
