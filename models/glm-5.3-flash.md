# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for Local Inference Lab.</em></p>

Serve `local-inference-lab/GLM-5.3-Flash-NVFP4` through the `glm53-flash`
profile in the [shared vLLM Docker guide](../docs/unified-vllm-docker.md).
The same image serves Qwen and DeepSeek; a model-specific image or entrypoint
is not required. The shared guide owns the image tag, launch command, LMCache
configuration and general option reference.

Status: **implemented** profile. **Qualified** bounded TP4/DCP1 MTP3 and
DFlash2 measurements are recorded below with their actual image identities.
The published beta's packaging tests do not imply a repeated full serving or
external-cache matrix on that exact registry digest.

## Start the server

Select `LIL_IMAGE` from the [image section](../docs/unified-vllm-docker.md#select-the-image),
then use these values in the [common launch command](../docs/unified-vllm-docker.md#start-a-server):

```bash
PROFILE=glm53-flash
GPU_DEVICES=0,1,2,3
TP=4
PORT=8000
SERVE_ARGS=(--mode mtp --draft-tokens 3)
```

Choose one speculation setting before running that command:

```bash
SERVE_ARGS=(--mode off)
```

```bash
SERVE_ARGS=(--mode mtp --draft-tokens 3)
```

```bash
SERVE_ARGS=(--mode dflash2 --draft-tokens 7)
```

DFlash2 downloads `local-inference-lab/GLM-5.3-Flash-DFlash2`, an offline
MXFP8 draft checkpoint, through the shared Hugging Face volume. It is not
online weight quantization. No absolute host model paths are required.

The API model name is `GLM-5.3-Flash-NVFP4`. Clients requiring the name
`GLM-5.3-Flash` can append `--served-model-name GLM-5.3-Flash` to `SERVE_ARGS`.
Changing the API name does not change the checkpoint.

## Serving defaults and alternatives

| Setting | Profile behavior |
|---|---|
| Parallelism | TP4/DCP1; four 96-GB GPUs in the measured configuration |
| Speculation when omitted | Off; the explicit examples select MTP3 or DFlash2 K7 |
| Target precision | ModelOpt NVFP4; B12X MoE and dense backends |
| Attention | B12X sparse attention/selection; FlashKDA recurrent prefill |
| MTP | B12X attention, Marlin draft MoE, private NVFP4 draft vocabulary head; BF16 target head |
| DFlash2 | Offline MXFP8 weights, B12X dense path, FLASH_ATTN draft attention, automatic draft KV dtype |
| Target KV | FP8; `--kv-cache-dtype nvfp4_ds_mla` is an explicit, separately unqualified option for this artifact |
| Graphs | Full-and-piecewise target/draft decode graphs, capture sizes through 256 |
| Scheduler | 4096 tokens, 32 sequences, one prefill lane, compute share 0.4 |
| Context / GPU fraction | 1,048,576 configured tokens / 0.93; configuration is not a million-token test |
| Prefix policy | `request_boundaries` with `mamba-cache-mode=align`; no manual retention interval needed |
| Sampling / reasoning | Temperature 1, top-p .95, reasoning `high`, `clear_thinking=false` |
| Vision | No artificial one/two-image profile cap; native encoder/context/memory limits remain |

Explicit request sampling and template options override their corresponding
server defaults. If replacing the complete template-default JSON, retain
`clear_thinking=false` when preserved assistant reasoning is required.

The [shared cache section](../docs/unified-vllm-docker.md#cache-storage-gpu-lmcache-or-native-offload)
documents GPU-local, LMCache RAM/filesystem and native KV offload. LMCache is
opt-in; the published image's 91 cache contract tests are not a model-level
RAM/restart-filesystem performance result.

`--decode-context-parallel-size 4` selects DCP4 and automatic full-CKV gather;
this is not specific to DFlash2. DCP4 and TP8 are **implemented**, but the
six-profile wheel comparison below qualifies DCP1, not those alternatives.
The [Spark TP2 recipe](glm-5.3-flash-spark-tp2.md) has a different checkpoint
and capacity contract; it is not this four-GPU profile with TP changed to two.

## Measured performance

Stock RTX PRO 6000 Workstation quartet, TP4/DCP1, 4096-token budget, GPU-only
FP8 target cache, full-and-piecewise graphs, temperature 1/top-p .95. Decode:
context zero, medians of three warmed 30-second runs. Prefill: sustained
uncached nominal-32K requests, client time to first token. C8 is aggregate.
Image identities, runtime arguments and raw samples are in the
[wheel-image qualification](../benchmarks/prepared-b12x-serving/).

| Mode | C1 tok/s, R35 → wheel image | C8 tok/s, R35 → wheel image | 32K prefill tok/s, R35 → wheel image | Sieve tok/s, R35 → wheel image |
|---|---:|---:|---:|---:|
| MTP3 | 247.12 → 249.33 (+0.89%) | 872.67 → 875.86 (+0.37%) | 15,332 → 15,580 (+1.62%) | 326.15 → 325.75 (−0.12%) |
| DFlash2 K7 | 211.23 → 219.83 (+4.07%) | 673.26 → 717.84 (+6.62%) | 15,538 → 15,734 (+1.26%) | 458.14 → 461.42 (+0.72%) |

Sieve uses five measured requests and is not a coding-correctness evaluation.
No no-spec measurement exists in this wheel-image matrix. Historical no-spec,
DCP4 and +6000-clock values remain in the archive, not substituted here.

### Prepared B12X plan integration

A separate same-GPU component comparison qualifies the B12X master
reconciliation and vLLM #789, which retains GLM's prepared selection plan:

| Metric | Control → prepared-plan image | Change |
|---|---:|---:|
| MTP3 C8 output | 856.05 → 857.27 tok/s | +0.14% |
| MTP3 32K prefill | 15,486 → 15,531 tok/s | +0.29% |
| DFlash2 C1 output | 214.31 → 212.08 tok/s | −1.04% |
| DFlash2 C8 output | 695.88 → 708.64 tok/s | +1.83% |
| DFlash2 32K prefill | 15,702 → 15,702 tok/s | 0.00% |

DFlash C1 verifier throughput rises 0.22% while acceptance changes; negative
output deltas are retained rather than called zero regression. The MTP control
contains C8 only. Read the
[component receipts](../benchmarks/prepared-b12x-contracts/#retained-pooled-selection-glm)
separately from the whole-image table. Two-shot is disabled in both comparisons.

Functional checks include arithmetic, repeated/changed prefix requests,
2/8/16 small images and image history. These are bounded checks, not general
language quality, arbitrary image resolution or million-token qualification.

## Quality evaluation and historical releases

Runtime throughput does not establish checkpoint quality. Retain the exact
runtime/checkpoint boundaries of these independent reports:

- [BF16, published NVFP4 and QAD AA-LCR comparison](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md)
  and [reproduction method](glm-5.3-flash/aa-lcr-reproduction.md).
- [Verifier-backed behavioral fidelity](glm-5.3-flash/verifier-backed-behavioral-fidelity.md),
  [QAD step-2500](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
  and [QAD TV-nucleus](glm-5.3-flash/qad-tvn-step2500-verifier-backed-behavioral-fidelity.md).
- [BF16/NVFP4 distribution fidelity](../kld/glm-5.3-flash-bf16-nvfp4.md)
  and [QAD quantization reports](../kld/glm-5.3-flash-qad-step2500.md).
- [Community R35 deployment and measurement archive](glm-5.3-flash-community-r35.md):
  release-specific launchers, DCP and no-spec matrices, +6000 measurements,
  source locks, historical LMCache restores and reported constrained-output limits.

Source review and unresolved items: [issue #773](https://github.com/local-inference-lab/vllm/issues/773).
