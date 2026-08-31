# GLM-5.1 Teacher-Forced Decode KLD / JS Reproduction

Measured on 2026-05-25 on the local 8x RTX PRO 6000 Blackwell host.

This page documents the decode-logprob KLD/JS test used for GLM-5.1
NVFP4/W4A16 and mixed FP8 checkpoints. It is intended to be reproducible from
the uploaded BF16 reference logits and the helper scripts in the Hugging Face
dataset below.

## What This Measures

The test compares full-vocabulary next-token probability distributions between
a BF16/source model and a quantized/model variant on the same decode prefix.

It is a teacher-forced decode test:

```text
source trajectory: T0, T1, T2, ...

step 0:
  variant sees prompt
  score variant logits for next token
  force sampled token to T0

step 1:
  variant sees prompt + T0
  score variant logits for next token
  force sampled token to T1

step 2:
  variant sees prompt + T0 + T1
  score variant logits for next token
  force sampled token to T2
```

The model is therefore only allowed to generate one token at a time and is then
snapped back to the BF16/source token. The reported KLD does not accumulate
autoregressive free-run drift. It answers this narrower question:

```text
On the BF16/source trajectory, how close is the variant's next-token
distribution to BF16?
```

It does not answer this separate question:

```text
If the variant free-runs for 30k or 100k tokens, can small local differences
move it onto a different future prefix trajectory?
```

For long-run quality, this test should be treated as a local distribution sanity
check, not as a substitute for free-run rollouts or task-level evals.

## Metrics

The primary teacher-forced metric is:

```text
KL(BF16/source || variant)
```

This is the natural direction because BF16/source defines the trajectory being
evaluated. Reverse KL can be computed because both full-vocab distributions are
available, but it is not the main teacher-forced metric because the variant is
not allowed to follow its own trajectory.

The symmetric metric reported here is Jensen-Shannon divergence:

```text
M = 0.5 * (P + Q)

JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

where:

```text
P = BF16/source next-token distribution
Q = variant next-token distribution
```

Lower is better. Zero means identical full-vocab distributions.

## Published Artifacts

Hugging Face dataset:

```text
https://huggingface.co/datasets/festr2/glm51-decode-kld-refs
```

Reference logits:

```text
refs/decode_teacher_bf16_ref_ctx2048_t17_20260525_085332.safetensors
refs/decode_teacher_bf16_ref_ctx2048_t17_20260525_085332.safetensors.json

refs/decode_teacher_multi_bf16_ref_p8_ctx2048_t64_20260525_220000.safetensors
refs/decode_teacher_multi_bf16_ref_p8_ctx2048_t64_20260525_220000.safetensors.json
```

Helper scripts:

```text
scripts/decode_logprob_kld.py
scripts/decode_logprob_kld_multi.py
scripts/teacher_force_logits_processor.py
```

The `teacher_force_logits_processor.py` processor forces the sampled token to
the source token after scoring. In vLLM V1 the returned `completion.logprobs`
are recorded before this custom logits processor is applied, so the saved
full-vocab logprobs remain the model's real prediction distribution, not the
forced one-hot sampling distribution.

## Tested Runtime

The 2026-05-25 measurements below used this local image:

```text
glm51-v4-explicit-w13-cleanup:test
sha256:58ad4e3dbae863a92f900a7c401a4de200d3d2add18d5ff8df333b07cfadd4d1
```

Source revisions embedded in the image labels:

```text
vLLM repo:   https://github.com/voipmonitor/vllm.git
vLLM branch: codex/explicit-w13-order-cleanup-20260525
vLLM commit: 356fc541268ebbd5529e349896d3eaf66e736382

B12X repo:   https://github.com/voipmonitor/b12x.git
B12X branch: codex/explicit-w13-order-cleanup-20260525
B12X commit: f6abdd287994141712f8401645afcc3e4b25dbc8
```

The public v4 production image documented in
[`../glm5.1_v4.md`](../glm5.1_v4.md) is the closest DockerHub baseline. For
exact result reproduction, use a container built from the source revisions
above.

## Models

BF16/source model:

```text
/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
```

NVFP4-MTP model:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

Mixed FP8PBWO L42-62 model:

```text
/root/.cache/huggingface/hub/models--festr2--glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517/snapshots/e37f1787435d2b2c111a5f5eac924a556a06e257
```

## Setup

Install or use an environment with `huggingface_hub`, then download the
published reference logits and scripts:

```bash
export KLD_ROOT=/root/kld/glm51-decode-kld-20260525
mkdir -p "${KLD_ROOT}"

huggingface-cli download festr2/glm51-decode-kld-refs \
  --repo-type dataset \
  --local-dir "${KLD_ROOT}" \
  --include 'refs/*' 'scripts/*'
```

Set the shared paths:

```bash
export IMAGE=glm51-v4-explicit-w13-cleanup:test
export CACHE_ROOT=/root/.cache/vllm-glm51-v4-kld

export BF16_MODEL=/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
export NVFP4_MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
export MIXED_FP8_MODEL=/root/.cache/huggingface/hub/models--festr2--glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517/snapshots/e37f1787435d2b2c111a5f5eac924a556a06e257

mkdir -p \
  "${CACHE_ROOT}/cutlass_dsl" \
  "${CACHE_ROOT}/jit" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/torchinductor" \
  "${CACHE_ROOT}/vllm"
```

Common Docker arguments:

```bash
COMMON_DOCKER_ARGS=(
  --rm
  --gpus all
  --ipc=host
  --network host
  --privileged
  --entrypoint /bin/bash
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -e OMP_NUM_THREADS=16
  -e SAFETENSORS_FAST_GPU=1
  -e CUTE_DSL_ARCH=sm_120a
  -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_IB_DISABLE=1
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  -e VLLM_USE_B12X_SPARSE_INDEXER=1
  -e VLLM_B12X_MLA_EXTEND_MAX_CHUNKS=32
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0
  -e VLLM_MTP_RETURN_NORMALIZED_HIDDEN=1
  -e VLLM_SPEC_ACCEPT_THRESHOLD_ACC=1.0
  -e VLLM_SPEC_ACCEPT_THRESHOLD_SINGLE=1.0
  -e VLLM_DEBUG_SYNC_MODEL_FORWARD=0
  -e VLLM_DEBUG_SYNC_SAMPLE_TOKENS=0
  -e VLLM_DEBUG_B12X_MLA=0
  -e VLLM_B12X_SYNC_DEBUG=0
  -e VLLM_B12X_MLA_DEBUG_FILE=
  -e PYTHONPATH=/workspace
  -e HF_HOME=/root/.cache/huggingface
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
  -e XDG_CACHE_HOME=/cache/jit
  -e CUDA_CACHE_PATH=/cache/jit
  -e VLLM_CACHE_DIR=/cache/jit/vllm
  -e TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer
  -e VLLM_CACHE_ROOT=/root/.cache/vllm
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl
  -v /root/.cache/huggingface:/root/.cache/huggingface
  -v "${CACHE_ROOT}/cutlass_dsl":/root/.cache/cutlass_dsl
  -v "${CACHE_ROOT}/jit":/cache/jit
  -v "${CACHE_ROOT}/triton":/root/.cache/triton
  -v "${CACHE_ROOT}/torchinductor":/root/.cache/torchinductor
  -v "${CACHE_ROOT}/vllm":/root/.cache/vllm
  -v "${KLD_ROOT}":/kld
  -v "${KLD_ROOT}/scripts/decode_logprob_kld.py":/workspace/decode_logprob_kld.py:ro
  -v "${KLD_ROOT}/scripts/decode_logprob_kld_multi.py":/workspace/decode_logprob_kld_multi.py:ro
  -v "${KLD_ROOT}/scripts/teacher_force_logits_processor.py":/workspace/teacher_force_logits_processor.py:ro
)
```

Important: do not set `NCCL_GRAPH_FILE=` for the patched NCCL image. Leave it
unset.

## Reuse Published BF16 References

For normal reproduction, use the published BF16 references instead of
regenerating them. Regenerating BF16 is slow because the full BF16 GLM-5.1
checkpoint requires CPU offload.

Single-prompt reference:

```bash
export BF16_SINGLE=/kld/refs/decode_teacher_bf16_ref_ctx2048_t17_20260525_085332.safetensors
```

This file has:

```text
prompt_len = 2048
max_tokens = 17
reported positions = 1..16 after --skip-prefill-next 1
```

Multi-prompt reference:

```bash
export BF16_MULTI=/kld/refs/decode_teacher_multi_bf16_ref_p8_ctx2048_t64_20260525_220000.safetensors
```

This file has:

```text
num_prompts = 8
prompt_len = 2048
max_tokens = 64
reported positions = 1..63 for each prompt after --skip-prefill-next 1
total reported positions = 504
```

## Optional: Regenerate BF16 Single Reference

Only do this when changing the source model, prompt generation, tokenizer, or
script behavior.

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" -lc '
  /opt/venv/bin/python /workspace/decode_logprob_kld.py collect \
    --label bf16_teacher_decode_ref \
    --model "'"${BF16_MODEL}"'" \
    --output /kld/refs/decode_teacher_bf16_ref_ctx2048_t17_new.safetensors \
    --prompt-len 2048 \
    --max-tokens 17 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --load-format auto \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --quantization none \
    --attention-backend B12X_MLA_SPARSE \
    --moe-backend auto \
    --cpu-offload-gb 110 \
    --teacher-force \
    --enforce-eager \
    --disable-custom-all-reduce
'
```

## Optional: Regenerate BF16 Multi Reference

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" -lc '
  /opt/venv/bin/python /workspace/decode_logprob_kld_multi.py collect \
    --label bf16_ref_p8_ctx2048_t64 \
    --model "'"${BF16_MODEL}"'" \
    --output /kld/refs/decode_teacher_multi_bf16_ref_p8_ctx2048_t64_new.safetensors \
    --num-prompts 8 \
    --prompt-len 2048 \
    --max-tokens 64 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --load-format auto \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --quantization none \
    --attention-backend B12X_MLA_SPARSE \
    --moe-backend auto \
    --cpu-offload-gb 110 \
    --teacher-force \
    --enforce-eager \
    --disable-custom-all-reduce
'
```

## Single-Prompt 16-Position Test

Collect NVFP4 A4:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=0 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=0 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld.py collect \
      --label nvfp4mtp_a4_single \
      --model "'"${NVFP4_MODEL}"'" \
      --output /kld/nvfp4mtp_a4_single.safetensors \
      --prompt-len 2048 \
      --max-tokens 17 \
      --token-source "'"${BF16_SINGLE}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.75 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Collect NVFP4 A16:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=1 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld.py collect \
      --label nvfp4mtp_a16_single \
      --model "'"${NVFP4_MODEL}"'" \
      --output /kld/nvfp4mtp_a16_single.safetensors \
      --prompt-len 2048 \
      --max-tokens 17 \
      --token-source "'"${BF16_SINGLE}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.75 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Collect mixed FP8PBWO L42-62 A16:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=1 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld.py collect \
      --label mixedfp8_l4262_a16_single \
      --model "'"${MIXED_FP8_MODEL}"'" \
      --output /kld/mixedfp8_l4262_a16_single.safetensors \
      --prompt-len 2048 \
      --max-tokens 17 \
      --token-source "'"${BF16_SINGLE}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.83 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Compare a single-prompt variant:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" -lc '
  /opt/venv/bin/python /workspace/decode_logprob_kld.py compare \
    --a "'"${BF16_SINGLE}"'" \
    --b /kld/nvfp4mtp_a4_single.safetensors \
    --output /kld/nvfp4mtp_a4_single_vs_bf16.json \
    --skip-prefill-next 1
'
```

Interpretation:

```text
--skip-prefill-next 1 means position 0 is excluded.
The reported single-prompt test uses 16 scored decode positions: 1..16.
```

## Multi-Prompt 8x64 Test

The multi-prompt test uses the same logic but scores 8 prompts and 64 generated
decode steps per prompt. With `--skip-prefill-next 1`, it reports 63 positions
per prompt, 504 positions total.

Collect NVFP4 A4:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=0 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=0 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld_multi.py collect \
      --label nvfp4mtp_a4_p8_ctx2048_t64 \
      --model "'"${NVFP4_MODEL}"'" \
      --output /kld/nvfp4mtp_a4_p8_ctx2048_t64.safetensors \
      --num-prompts 8 \
      --prompt-len 2048 \
      --max-tokens 64 \
      --token-source "'"${BF16_MULTI}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.75 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Collect NVFP4 A16:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=1 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld_multi.py collect \
      --label nvfp4mtp_a16_p8_ctx2048_t64 \
      --model "'"${NVFP4_MODEL}"'" \
      --output /kld/nvfp4mtp_a16_p8_ctx2048_t64.safetensors \
      --num-prompts 8 \
      --prompt-len 2048 \
      --max-tokens 64 \
      --token-source "'"${BF16_MULTI}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.75 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Collect mixed FP8PBWO L42-62 A16:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e VLLM_B12X_FORCE_MOE_A16=1 \
  -e B12X_MOE_FORCE_A16=0 \
  -e VLLM_B12X_MOE_DECODE_A16=1 \
  "${IMAGE}" -lc '
    /opt/venv/bin/python /workspace/decode_logprob_kld_multi.py collect \
      --label mixedfp8_l4262_a16_p8_ctx2048_t64 \
      --model "'"${MIXED_FP8_MODEL}"'" \
      --output /kld/mixedfp8_l4262_a16_p8_ctx2048_t64.safetensors \
      --num-prompts 8 \
      --prompt-len 2048 \
      --max-tokens 64 \
      --token-source "'"${BF16_MULTI}"'" \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.83 \
      --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --load-format fastsafetensors \
      --max-model-len 4096 \
      --max-num-batched-tokens 2048 \
      --quantization modelopt_fp4 \
      --attention-backend B12X_MLA_SPARSE \
      --moe-backend b12x \
      --teacher-force \
      --enforce-eager \
      --disable-custom-all-reduce
  '
```

Compare a multi-prompt variant:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" -lc '
  /opt/venv/bin/python /workspace/decode_logprob_kld_multi.py compare \
    --a "'"${BF16_MULTI}"'" \
    --b /kld/nvfp4mtp_a4_p8_ctx2048_t64.safetensors \
    --output /kld/nvfp4mtp_a4_p8_ctx2048_t64_vs_bf16.json \
    --skip-prefill-next 1
'
```

## Expected Results From 2026-05-25

Single prompt, 16 scored decode positions:

```text
NVFP4-MTP A4:
  KL(BF16 -> variant): 1.305212e-05
  reverse KL:          1.914381e-05
  JS mean:             3.679768e-06

NVFP4-MTP A16:
  KL(BF16 -> variant): 1.441972e-05
  reverse KL:          2.193232e-05
  JS mean:             4.113602e-06

mixed FP8 L42-62 A16:
  KL(BF16 -> variant): 4.892056e-05
  reverse KL:          7.004590e-05
  JS mean:             1.409706e-05
```

Eight prompts, 504 scored decode positions:

```text
NVFP4-MTP A4:
  KL(BF16 -> variant): 7.962753e-06
  reverse KL:          1.152314e-05
  JS mean:             2.151593e-06
  JS max:              1.787597e-04

NVFP4-MTP A16:
  KL(BF16 -> variant): 9.172449e-06
  reverse KL:          8.865960e-06
  JS mean:             2.154707e-06
  JS max:              8.195499e-05

mixed FP8 L42-62 A16:
  KL(BF16 -> variant): 1.897077e-05
  reverse KL:          2.755633e-05
  JS mean:             5.439697e-06
  JS max:              1.515283e-04
```

Interpretation of these specific numbers:

```text
NVFP4 A4 and NVFP4 A16 are very close to BF16 on this teacher-forced decode
trajectory. A16 is not divergent, but it does not clearly improve the mean
teacher-forced decode KLD versus A4 in this run.

Mixed FP8 L42-62 A16 remains low in absolute terms, but is worse than pure
NVFP4 A4/A16 on this decode distribution test.
```

This is not a final long-rollout quality verdict. It is a local distribution
check on BF16/source prefixes.

