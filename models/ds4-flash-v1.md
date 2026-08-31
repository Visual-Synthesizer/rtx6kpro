# DeepSeek-V4-Flash v1 on 8x RTX PRO 6000 Blackwell

Status: measured locally on 2026-06-05. This page records the current
`dev/nameless-ascent` B12X/vLLM image for DeepSeek-V4-Flash TP2/TP4 and the
first clean MTP-on/off decode sweep.

## Image

```bash
voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605
```

Image ID:

```text
sha256:0dbf3ecffe7ea69874b107169a33ce70a558db1896a6e14d56a19c48a9d613a8
```

Image metadata:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS | `13.4.1.2-1` |
| cuDNN | `9.22.0.52-1` |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical` |
| PyTorch | `2.12.0+cu132` |
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `codex/nameless-ascent-66b2a76-upstream-pr1-20260605` |
| vLLM commit | `05cc7ba06` |
| Luke branch source | `https://github.com/local-inference-lab/vllm/tree/dev/nameless-ascent` at `66b2a7688c753b160a2856f41e069560fddce8fb` |
| B12X repo | `https://github.com/lukealonso/b12x.git` |
| B12X commit | `60a63d8cc5cb9eb5022304af79e6abc5c2cca576` |
| FlashInfer repo | `https://github.com/flashinfer-ai/flashinfer.git` |
| FlashInfer commit | `e8d31317bedb4efd52559a2234f4cb9e83428cb9` |

Important fixes in this image:

| Area | Fix |
|---|---|
| DS4 MTP | `serve-ds4-flash.sh` defaults to `use_local_argmax_reduction:false`; the old `true` setting produced corrupted short outputs and CJK leakage. |
| Loader | DS4 Flash defaults to `--load-format safetensors`; current public snapshot has safetensors shards, not usable InstantTensor artifacts. |
| DCP sparse MLA | DCP sparse MLA disables full CUDA graph replay and uses piecewise graphs to avoid long-context corruption for DCP>1. |

Verify:

```bash
IMAGE=voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605

docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -m json.tool
docker run --rm "$IMAGE" bash -lc \
  'grep -n use_local_argmax_reduction /opt/vllm/serve-ds4-flash.sh;
   /opt/vllm/.venv/bin/python -c "import importlib.metadata as md; [print(p, md.version(p)) for p in (\"vllm\",\"torch\",\"flashinfer-python\",\"b12x\")]"'
```

## Runtime

Default launcher:

```text
/opt/vllm/serve-ds4-flash.sh
```

Default model:

```text
deepseek-ai/DeepSeek-V4-Flash
```

Default MTP config when `VLLM_ENABLE_MTP=1`:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 2,
  "draft_sample_method": "probabilistic",
  "moe_backend": "b12x",
  "use_local_argmax_reduction": false
}
```

The spec config can be overridden without editing the image:

```bash
DS4_SPEC_CONFIG_JSON='{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":false}'
```

## Docker Run

```bash
cat >/root/run-ds4-flash-v1 <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:cu132-vllm05cc7ba-b12x60a63d8-fi-e8d3131-ds4glm-20260605}"
NAME="${NAME:-ds4-flash-v1}"
PORT="${PORT:-5329}"
TP_SIZE="${TP_SIZE:-4}"
MTP="${MTP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.875}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-140000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-4096}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

exec docker run -d --gpus all \
  --name "${NAME}" \
  -p "${PORT}:${PORT}" \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -e PORT="${PORT}" \
  -e TP_SIZE="${TP_SIZE}" \
  -e DCP_SIZE=1 \
  -e VLLM_ENABLE_MTP="${MTP}" \
  -e LOAD_FORMAT=safetensors \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  "${IMAGE}" \
  bash -lc 'cd /opt/vllm && ./serve-ds4-flash.sh'
EOF
chmod +x /root/run-ds4-flash-v1
```

Examples:

```bash
# TP4 + MTP, default benchmark profile.
TP_SIZE=4 MTP=1 PORT=5329 /root/run-ds4-flash-v1

# TP4 no-MTP.
TP_SIZE=4 MTP=0 PORT=5329 /root/run-ds4-flash-v1

# TP2 + MTP.
TP_SIZE=2 MTP=1 PORT=5329 /root/run-ds4-flash-v1
```

Readiness:

```bash
curl -fsS http://127.0.0.1:5329/v1/models | jq .
docker logs ds4-flash-v1 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Graph capturing finished|Application startup complete' | tail -20
```

## Correctness Smoke

`python3 /mnt/test.py --port 5329` and
`python3 /mnt/test.py --port 5329 -c10000` were run for TP2/TP4 MTP on/off.
All four combinations returned coherent English output with `0` CJK characters
after disabling local argmax reduction for MTP.

| Profile | Short gen tok/s | 10k-context gen tok/s | CJK |
|---|---:|---:|---:|
| TP2 no-MTP | 108.7 | 115.6 | 0 |
| TP2 MTP | 203.3 | 207.4 | 0 |
| TP4 no-MTP | 126.7 | 136.3 | 0 |
| TP4 MTP | 250.0 | 257.6 | 0 |

The broken configuration was:

```json
{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}
```

It reproduced corrupted Python output and CJK leakage on short smoke tests.

## Full Decode Sweep

Final full sweep settings:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5329 \
  --model DeepSeek-V4-Flash \
  --contexts 0,16k,32k,64k,128k \
  --concurrency 1,2,4,8,16 \
  --duration 30 \
  --dcp-size 1 \
  --kv-budget KV_TOKENS \
  --token-targeting exact \
  --max-tokens 8192 \
  --display-mode plain \
  --output OUT.json
```

TP2 was also probed with `32,64,128` concurrency; values above `MAX_NUM_SEQS=16`
are capacity-limited and hidden in the headline. `MAX_MODEL_LEN=140000` is
required because benchmark `128k` is `131,072` prompt tokens and the request
also reserves `max_tokens=8192`.

Results are under:

```text
/root/bench-results/ds4-flash-v1-full-exact-140k-20260605/
```

### Prefill Scout Speed

Client prompt tok/s from integrated scout requests:

| Profile | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| TP2 no-MTP | 6,754 | 6,686 | 6,568 | 6,320 | 5,856 |
| TP2 MTP | 6,529 | 6,411 | 6,283 | 6,043 | 5,605 |
| TP4 no-MTP | 8,177 | 8,132 | 7,964 | 7,621 | 7,001 |
| TP4 MTP | 7,962 | 7,850 | 7,639 | 7,334 | 6,757 |

### TP2 No-MTP

Aggregate decode tok/s:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 117.5 | 198.3 | 322.7 | 482.7 | 711.3 | capacity-limited | ∅ | ∅ |
| 16k | 116.5 | 193.6 | 307.8 | 451.2 | 625.5 | ∅ | ∅ | ∅ |
| 32k | 116.2 | 192.1 | 306.9 | 441.2 | ∅ | ∅ | ∅ | ∅ |
| 64k | 113.8 | 189.4 | 301.5 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 111.0 | 181.6 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

### TP2 MTP

Aggregate decode tok/s:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 190.0 | 300.6 | 386.0 | 644.0 | 873.3 | capacity-limited | ∅ | ∅ |
| 16k | 191.3 | 228.7 | 375.2 | 543.8 | ∅ | ∅ | ∅ | ∅ |
| 32k | 203.3 | 272.7 | 365.1 | 518.9 | ∅ | ∅ | ∅ | ∅ |
| 64k | 186.8 | 221.6 | 365.2 | ∅ | ∅ | ∅ | ∅ | ∅ |
| 128k | 162.3 | 279.1 | ∅ | ∅ | ∅ | ∅ | ∅ | ∅ |

Average server speculative acceptance:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 0 | 76.6% | 68.7% | 52.4% | 72.2% | 67.4% |
| 16k | 74.0% | 46.4% | 64.0% | 62.7% | ∅ |
| 32k | 85.8% | 61.7% | 59.0% | 61.0% | ∅ |
| 64k | 59.2% | 29.2% | 44.5% | ∅ | ∅ |
| 128k | 72.6% | 73.2% | ∅ | ∅ | ∅ |

### TP4 No-MTP

Aggregate decode tok/s:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 0 | 140.3 | 250.6 | 422.7 | 664.9 | 991.9 |
| 16k | 139.1 | 247.9 | 408.2 | 622.6 | 894.3 |
| 32k | 138.7 | 242.9 | 403.4 | 617.3 | 869.4 |
| 64k | 135.4 | 237.7 | 391.4 | 592.7 | 825.2 |
| 128k | 131.2 | 226.2 | 361.5 | 536.5 | 735.2 |

### TP4 MTP

Aggregate decode tok/s:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 0 | 240.7 | 414.2 | 446.5 | 712.1 | 384.3 |
| 16k | 249.2 | 405.4 | 443.0 | 555.4 | 744.9 |
| 32k | 238.4 | 410.2 | 489.6 | 567.8 | warmup-limited |
| 64k | 245.2 | 323.3 | 350.4 | 502.8 | warmup-limited |
| 128k | 216.4 | warmup-limited | 469.8 | 516.8 | warmup-limited |

Average server speculative acceptance:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| 0 | 54.8% | 72.1% | 42.3% | 28.9% | 0.0% |
| 16k | 83.5% | 73.8% | 34.1% | 14.9% | 0.0% |
| 32k | 76.0% | 76.9% | 62.7% | 15.6% | warmup-limited |
| 64k | 79.0% | 58.8% | 24.1% | 16.0% | warmup-limited |
| 128k | 78.0% | warmup-limited | 73.2% | 30.4% | warmup-limited |

Raw JSON:

```text
/root/bench-results/ds4-flash-v1-full-exact-140k-20260605/tp2-mtp0-full.json
/root/bench-results/ds4-flash-v1-full-exact-140k-20260605/tp2-mtp1-full.json
/root/bench-results/ds4-flash-v1-full-exact-140k-20260605/tp4-mtp0-full-c1-16.json
/root/bench-results/ds4-flash-v1-full-exact-140k-20260605/tp4-mtp1-full-c1-16.json
```

## Notes

`cc32+` is not a useful headline for this profile because `MAX_NUM_SEQS=16`.
TP4 was therefore rerun as a clean `c=1,2,4,8,16` matrix after confirming that
cc32/64/128 only exercise queue detection.

TP4 MTP is not uniformly better. It is strong at cc1/cc2, but several long or
high-concurrency cells show speculative acceptance collapse. The worst case in
the server log was repeated `0.0%` acceptance at cc16, so use TP4 no-MTP as the
safer production default until this is understood.

The server logs show B12X PCIe oneshot allreduce for TP groups and PYNCCL for
EP groups. FlashInfer autotune is enabled, but this DS4 B12X path logs that no
FlashInfer compute kernels are active during the warmup.
