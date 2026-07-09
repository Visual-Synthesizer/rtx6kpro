# Kimi-K2.7-Code v3 on Fathomless

This page documents the Kimi-K2.7-Code DFlash validation on the shared
Fathomless Firmament image plus the Kimi runtime patch from vLLM PR #86.

## Image And Patch

Validated base image:

```text
voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709
voipmonitor/vllm@sha256:2dbc40a1fd104168226f46eb31f14301967a37aca95fed71fd23ff4f74b10698
```

Patch overlay:

```text
https://github.com/local-inference-lab/vllm/pull/86
vLLM branch: codex/ff-kimi-k27-dflash-v3-20260709
vLLM commit: cc584145a6d81844fa0fb78078f2648eabe2b30f
Base branch: dev/fathomless-firmament @ 4cf20be8682749d0cca18639304a1693b00ce421
```

The validation below used the base image with PR #86 files mounted over the
installed Python package. After PR #86 is merged and a new image is built, the
bind mounts can be removed.

## Models

Target:

```text
moonshotai/Kimi-K2.7-Code
```

DFlash draft:

```text
/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a
```

## Runtime

| Setting | Value |
|---|---|
| TP / DCP | `8 / 4` |
| Target attention | `TRITON_MLA` |
| Draft attention | `TRITON_ATTN` |
| KV cache | `fp8` |
| Runner | V2 |
| DFlash tokens | `7` |
| Tool parser | `kimi_k2` |
| Reasoning parser | `kimi_k2` |
| Custom allreduce | disabled |
| Kimi DCP4 compatibility env | `VLLM_DFLASH_FORCE_SINGLE_SWA=1` |

With DFlash `num_speculative_tokens=7`, CUDA graph capture sizes must include a
multiple of `8`. The debug validation below used `--max-num-seqs 1` and
`--max-cudagraph-capture-size 8`.

## Overlay Docker Run

```bash
IMAGE=voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709
PATCH_ROOT=/root/vllm/worktrees/vllm-fathomless-v15
DRAFT=/root/.cache/huggingface/hub/models--SubSir--Kimi-K2.6-DFlash-tmp/snapshots/171a2d3e68ec4050abe66c298477056b2fc2d40a

docker rm -f kimi-k27-code-v3-dcp4 2>/dev/null || true

docker run -d --init --name kimi-k27-code-v3-dcp4 \
  --network host --ipc=host --gpus '"device=0,1,2,3,4,5,6,7"' \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e VLLM_DFLASH_FORCE_SINGLE_SWA=1 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v "$PATCH_ROOT/vllm/model_executor/models/qwen3_dflash.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro" \
  -v "$PATCH_ROOT/vllm/v1/spec_decode/dflash.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/spec_decode/dflash.py:ro" \
  -v "$PATCH_ROOT/vllm/v1/worker/gpu_model_runner.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py:ro" \
  -v "$PATCH_ROOT/vllm/v1/attention/backends/mla/triton_mla.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/triton_mla.py:ro" \
  -v "$PATCH_ROOT/vllm/v1/attention/ops/triton_unified_attention.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro" \
  "$IMAGE" /opt/venv/bin/vllm serve moonshotai/Kimi-K2.7-Code \
    --served-model-name Kimi-K2.7-Code \
    --host 0.0.0.0 \
    --port 7801 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --decode-context-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --attention-backend TRITON_MLA \
    --gpu-memory-utilization 0.94 \
    --max-model-len 262144 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 8 \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode weights \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --enable-auto-tool-choice \
    --async-scheduling \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --speculative-config "{\"model\":\"$DRAFT\",\"method\":\"dflash\",\"num_speculative_tokens\":7,\"attention_backend\":\"TRITON_ATTN\"}"
```

Wait until:

```text
Starting vLLM server on http://0.0.0.0:7801
```

## Validation Commands

Run a small generation smoke first. The loop was stopped manually after seven
clean iterations.

```bash
python3 /mnt/test.py --port 7801 -L
```

Run prefill before the final decode check. The first decode immediately after
boot may include Triton JIT compilation, so warmed decode is the regression
signal.

```bash
python3 /root/llm_decode_bench.py \
  --port 7801 \
  --model Kimi-K2.7-Code \
  --prefill-only \
  --standalone-prefill \
  --prefill-contexts 8k,64k \
  --prefill-duration 10 \
  --display-mode plain \
  --output /root/bench-results/kimi-k27-v3-ff-dcp4-current-prefill-20260709/prefill_8k_64k.json

python3 /root/llm_decode_bench.py \
  --port 7801 \
  --model Kimi-K2.7-Code \
  --concurrency 1 \
  --contexts 0k \
  --duration 30 \
  --max-tokens 8192 \
  --skip-prefill \
  --display-mode plain \
  --output /root/bench-results/kimi-k27-v3-ff-dcp4-current-prefill-20260709/decode_cc1_after_prefill.json
```

## Results

Measured on GPUs `0-7`, TP8/DCP4, `max_num_seqs=1`, graph cap `8`.

| Test | Result |
|---|---:|
| KV cache budget, vLLM metrics | 1,604,736 tokens |
| DFlash7 warmed 0k cc1 decode | 139.3 tok/s |
| Standalone prefill 8k | 7,971 tok/s |
| Standalone prefill 64k | 4,918 tok/s |
| `test.py -L` smoke | 7 iterations, 5,807 tokens, CJK 0 |
| `test.py -L` average generation-only | 205.1 tok/s |

During the warmed decode, server metrics reached about `25-29%` average draft
acceptance in the steady part of the run. The relevant log intervals showed
generation throughput up to `146.9 tok/s`.

The first decode after boot is not a reliable regression signal for this profile
because Triton JIT warnings appeared for `_topk_topp_kernel`,
`_prepare_dflash_inputs_kernel`, `_fwd_grouped_kernel_stage1`, and
`_fwd_kernel_stage2`. Use a warmed decode run.

## Artifacts

```text
/root/bench-results/kimi-k27-v3-ff-dcp4-current-prefill-20260709/prefill_8k_64k.json
/root/bench-results/kimi-k27-v3-ff-dcp4-current-prefill-20260709/decode_cc1_after_prefill.json
```
