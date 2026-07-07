# GLM-5.2 Unsloth-Style Prefill KLD Reproduction

Measured on 2026-07-07 on the local 16x RTX PRO 6000 Blackwell host.

This page documents the vLLM reproduction of the Unsloth-style KLD check used
for GLM-5.2. It is intentionally separate from the regular GLM-5.2 v14 KLD
tables because this run uses a different corpus, chunking rule, context length,
and aggregation style.

The goal is reproducibility: the BF16 reference logprobs are published, the
exact scripts are published with them, and the measured result is included
below.

## Published Artifacts

Hugging Face dataset:

```text
https://huggingface.co/datasets/festr2/GLM-5.2-Unsloth-Style-KLD-Refs-20260707
```

The dataset contains:

```text
reference-logprobs/meta.json
reference-logprobs/batch_*.safetensors
scripts/unsloth_style_prefill_kld_vllm.py
scripts/run_glm52_unsloth_style_a16_kld_20260707.sh
patches/0001-vllm-prompt-logits-export-for-kld.patch
results/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/
```

The reference logprob cache is large:

```text
files: 565 batch_*.safetensors files + meta.json
size: 167 GB
```

## What This Measures

This is a vLLM equivalent of the important parts of the Unsloth
`llama-perplexity` KLD logs:

```text
Reference corpus: Salesforce/wikitext, wikitext-2-raw-v1, split=test
Text join rule: keep blank rows, join rows with "\n"
Chunking: non-overlapping ctx512 chunks
Chunks: 565
Positions: 288,715
Compared distribution: full-vocabulary prompt-position distribution
Direction: KL(BF16/base || candidate)
EOS handling: mask EOS token and renormalize both distributions
```

The run compares local next-token distributions on the same fixed WikiText-2
token sequence. It does not measure free-run generation drift.

This run was motivated by the Unsloth public KLD logs:

```text
https://huggingface.co/unsloth/Qwen3.5-35B-A3B-Experiments-GGUF/tree/main/KLD_Logs
```

## Result

Candidate:

```text
lukealonso/GLM-5.2-NVFP4
snapshot: 8a1f4a13204acf2b7ac840375efaed64c231c522
mode: A16 force
DCP: 1
MTP: off
TP: 8
```

BF16 reference:

```text
zai-org/GLM-5.2
snapshot: 4d67f66cc64d3219133b767c253b2ad1425c6c88
TP: 16
```

Summary file:

```text
results/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/a16_dcp1_mtp0/summary.json
```

| Metric | Value |
|---|---:|
| Mean KLD | 0.0845515281 |
| KLD standard error | 0.0004281334 |
| Positions | 288,715 |
| Chunks | 565 |
| Same top token | 90.7591% |
| Mean PPL BF16/base | 3.632230 |
| Mean PPL candidate | 3.736037 |
| Mean PPL candidate / BF16 | 1.028579 |
| Mean ln(PPL candidate / BF16) | 0.0281786 |
| Target probability delta mean | -0.9381% |
| Target probability delta RMS | 10.6477% |

KLD percentiles:

| Percentile | KLD |
|---:|---:|
| 0.1 | 0.0000000437 |
| 1 | 0.0000008448 |
| 5 | 0.0000097069 |
| 10 | 0.0000393630 |
| 50 | 0.0141873630 |
| 90 | 0.2096127123 |
| 95 | 0.3885383904 |
| 99 | 1.0776491165 |
| 99.9 | 2.6957230568 |
| max | 11.5384511948 |

The recorded minimum KLD was `-0.000000194978`, which is floating-point noise
after EOS masking and renormalization.

## Tested Runtime

Docker image:

```text
voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
```

Embedded image revisions:

```text
vLLM repo:   https://github.com/local-inference-lab/vllm.git
vLLM branch: fable/dcp-b12x-contiguous-lse-20260707
vLLM commit: e2e2eaf61d05834fb5f7f529b75ce75c4cafc289

B12X branch: refs/pull/26/head
B12X commit: 26144c0eda970ce7e30bf7c64a2f094abe1fea4d

InstantTensor commit: 85e7c5f5539d9c006ee0c26bc1b5233c65251b6b
NCCL: local-inference 2.30.4
CUDA: 13.2.1
PyTorch: 2.12.0+cu132
```

vLLM version printed by the run:

```text
v0.11.2.dev279+eldritch.enlightenment.v7.vllme2e2eaf.b12x26144c0.cu132.20260707
```

## Prompt-Logits Overlay

The image above did not natively expose raw prompt logits through the public
vLLM output API. The run therefore mounted a small vLLM overlay that adds
`SamplingParams.return_prompt_logits` and `RequestOutput.prompt_logits`.

The dataset includes the exact overlay patch:

```text
patches/0001-vllm-prompt-logits-export-for-kld.patch
```

Overlay provenance:

```text
base repo: https://github.com/local-inference-lab/vllm.git
base branch: codex/dark-devotion-release-20260622
base commit: ec656676100a756912d6966c4232ea436c55d792
overlay commit: 73a005ca76a8be0dd085e7d07d0d581a57b4ebf5
```

Create the overlay checkout:

```bash
export LOGITS_OVERLAY=/root/vllm/worktrees/vllm-release-kld-logits-export-20260622
mkdir -p /root/vllm/worktrees

git clone https://github.com/local-inference-lab/vllm.git "${LOGITS_OVERLAY}"
cd "${LOGITS_OVERLAY}"
git checkout ec656676100a756912d6966c4232ea436c55d792
git am /root/kld/glm52-unsloth-style-kld-20260707/patches/0001-vllm-prompt-logits-export-for-kld.patch
```

The runner copies the overlay files into the container's site-packages before
starting the KLD script.

## Download Published References

Install or use an environment with `huggingface_hub`, then download the
published reference cache and scripts:

```bash
export KLD_ARTIFACT_ROOT=/root/kld/glm52-unsloth-style-kld-20260707
mkdir -p "${KLD_ARTIFACT_ROOT}"

huggingface-cli download festr2/GLM-5.2-Unsloth-Style-KLD-Refs-20260707 \
  --repo-type dataset \
  --local-dir "${KLD_ARTIFACT_ROOT}"

cp -a "${KLD_ARTIFACT_ROOT}/scripts/"* /root/kld/
chmod +x /root/kld/run_glm52_unsloth_style_a16_kld_20260707.sh
```

The download requires roughly 170 GB of local storage.

## Reproduce Candidate Comparison

This reuses the uploaded BF16 reference cache and only reruns the candidate
side. It is the fastest exact check.

```bash
export IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
export LOGITS_OVERLAY=/root/vllm/worktrees/vllm-release-kld-logits-export-20260622
export REF_DIR=/root/kld/glm52-unsloth-style-kld-20260707/reference-logprobs
export OUT_ROOT=/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_repro_$(date -u +%Y%m%dT%H%M%SZ)

/root/kld/run_glm52_unsloth_style_a16_kld_20260707.sh compare
```

Expected output path:

```text
${OUT_ROOT}/a16_dcp1_mtp0/summary.json
```

The result should be close to:

```text
mean_kld = 0.0845515281
kld_standard_error = 0.0004281334
positions = 288715
```

## Reproduce Full Run From Scratch

This regenerates the BF16 reference cache and then compares the candidate.
It uses all 16 GPUs for BF16 reference collection and GPUs 0-7 for the
candidate comparison.

```bash
export IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
export LOGITS_OVERLAY=/root/vllm/worktrees/vllm-release-kld-logits-export-20260622
export BF16_MODEL=/root/.cache/huggingface/hub/models--zai-org--GLM-5.2/snapshots/4d67f66cc64d3219133b767c253b2ad1425c6c88
export NVFP4_MODEL=/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
export REF_DIR=/dev/shm/glm52_unsloth_style_bf16_ctx512_ref_full
export OUT_ROOT=/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_$(date -u +%Y%m%dT%H%M%SZ)

/root/kld/run_glm52_unsloth_style_a16_kld_20260707.sh full
```

The original full run was:

```text
OUT_ROOT=/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z
REF_DIR=/dev/shm/glm52_unsloth_style_bf16_ctx512_ref_full
```

## Exact Runtime Settings

Common environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_DEVICE_MAX_CONNECTIONS=32
OMP_NUM_THREADS=16
CUTE_DSL_ARCH=sm_120a
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_USE_AOT_COMPILE=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_USE_MEGA_AOT_ARTIFACT=1
VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_MHC=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_ENABLE_PCIE_ALLREDUCE=1
B12X_MLA_SM120_UNIFIED=1
B12X_DENSE_SPLITK_TURBO=1
B12X_W4A16_TC_DECODE=1
NCCL_IB_DISABLE=1
NCCL_P2P_LEVEL=SYS
NCCL_PROTO=LL,LL128,Simple
VLLM_NCCL_SO_PATH=/opt/libnccl.so.2.30.4
LD_PRELOAD=/opt/libnccl.so.2.30.4
```

Index override:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

`hf_overrides`:

```json
{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}
```

BF16 reference pass:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
tensor_parallel_size=16
decode_context_parallel_size=1
dtype=bfloat16
kv_cache_dtype=fp8
load_format=safetensors
quantization=none
attention_backend=B12X_MLA_SPARSE
moe_backend=auto
max_model_len=1024
max_num_batched_tokens=512
max_num_seqs=1
gpu_memory_utilization=0.98
kv_cache_memory_bytes=1073741824
enforce_eager=true
disable_custom_all_reduce=true
B12X_MOE_FORCE_A8=0
B12X_MOE_FORCE_A16=0
```

Candidate comparison:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tensor_parallel_size=8
decode_context_parallel_size=1
dtype=bfloat16
kv_cache_dtype=fp8
load_format=instanttensor
quantization=modelopt_fp4
attention_backend=B12X_MLA_SPARSE
moe_backend=b12x
max_model_len=1024
max_num_batched_tokens=512
max_num_seqs=1
gpu_memory_utilization=0.86
kv_cache_memory_bytes=1073741824
enforce_eager=true
disable_custom_all_reduce=true
B12X_MOE_FORCE_A8=0
B12X_MOE_FORCE_A16=1
```

The full run uses `BATCH_PROMPTS=1`. This avoids hundreds of GB of extra
transient memory pressure while collecting full-vocabulary prompt logits.

## Implementation Notes

The comparison script computes KLD as:

```text
sum(exp(ref_logprob) * (ref_logprob - candidate_logprob))
```

Rows where the masked token has `ref_logprob=-inf` are handled explicitly to
avoid `0 * -inf = NaN`. Both reference and candidate distributions mask the EOS
token and are renormalized before comparison.

The reference pass writes full-vocabulary prompt logprobs to safetensors. The
candidate pass streams those reference tensors back from disk and compares them
against freshly generated candidate prompt logits.

The runner starts the BF16 and candidate phases sequentially. Do not start a
second model load while measuring the candidate phase if you want a clean
timing/debug log.

## Source Files From The Original Run

Local run artifacts:

```text
/root/kld/unsloth_style_prefill_kld_vllm.py
/root/kld/run_glm52_unsloth_style_a16_kld_20260707.sh
/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/config.txt
/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/collect_ref.log
/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/compare_a16.log
/root/kld/glm52_unsloth_style_a16_dcp1_mtp0_full_20260707T180811Z/a16_dcp1_mtp0/summary.json
```

The same scripts and result files are included in the published Hugging Face
dataset.
