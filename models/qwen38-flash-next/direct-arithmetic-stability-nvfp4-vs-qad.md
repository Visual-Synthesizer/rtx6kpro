# Qwen3.8-Flash-Next direct-answer arithmetic stability: NVFP4 versus QAD

Status: **qualified for the declared checkpoints, vLLM runtime, prompts, and
sampling cells**. The QAD checkpoint remains **research-only** as a deployment
target.

This report measures a narrow failure mode in which Qwen3.8-Flash-Next must
return only an integer while reasoning is disabled. It compares the published
NVIDIA 4-bit floating-point (NVFP4) checkpoint with the Local Inference Lab
quantization-aware-distillation (QAD) checkpoint under the same vLLM serving
configuration.

QAD strongly reduced errors on the source `99 × 17` reproducer, from
**25/4,200 to 2/4,200**, but did not eliminate them. On a separate suite of 600
unique arithmetic tasks, QAD improved exact accuracy with reasoning disabled
from **77.17% to 78.69%**, a difference of **+1.53 percentage points**. The
task-cluster bootstrap 95% interval is **+0.50 to +2.56 points**.

The result does **not** support the broader claim that QAD fixes arithmetic.
With low reasoning, both checkpoints scored approximately 99.6%, while both
failed most parenthesized two-operation tasks when reasoning was disabled. QAD
reduced the frequency of the measured failure mode; enabling low reasoning had
a much larger effect.

## Source `99 × 17` reproducer

The source prompt is:

```text
Case stability3-effort-76. Compute 99 times 17. Give only the integer as your final answer.
```

The correct answer is `1683`. The single-prompt reproducer is
[`arithmetic_probe.py`](https://gist.github.com/ktsaou/cd36c6bed6a4a1947ff15c632964dd2d),
revision `8e2d144ef0ac4287183cbc7a34b7ec22e3c993be`. Its SHA-256 is
`f1d39496647be574c7ed0cd2456c6fea0a12b04f2990f8e9ce4ff0329f9774e2`.

Every temperature/checkpoint cell contains 600 requests: 100 attempts in each
combination of Chat Completions, Responses, and Messages APIs with streaming
enabled and disabled. The 100 attempts per API/stream mode were divided across
eight replicas as 13 attempts on four replicas and 12 on four replicas. The
client allowed four concurrent requests per replica, 32 per checkpoint; both
checkpoint pools ran simultaneously. No request seed or retry was used.

| Temperature | Published NVFP4 | QAD |
|---:|---:|---:|
| 0.0 | 596/600 | 600/600 |
| 0.1 | 600/600 | 600/600 |
| 0.2 | 598/600 | 600/600 |
| 0.5 | 597/600 | 600/600 |
| 0.7 | 598/600 | 599/600 |
| 0.9 | 596/600 | 599/600 |
| 1.0 | 590/600 | 600/600 |
| **All cells** | **4,175/4,200 (99.405%)** | **4,198/4,200 (99.952%)** |

The 25 published-NVFP4 errors were `153` 22 times and `1503`, `1533`, and
`1513` once each. QAD returned `153` once and `1503` once. All 8,400 requests
completed without a request or protocol error.

| API surface | Published NVFP4 errors | QAD errors |
|---|---:|---:|
| Chat Completions | 14/1,400 | 1/1,400 |
| Responses | 11/1,400 | 1/1,400 |
| Messages | 0/1,400 | 0/1,400 |

The API rows are diagnostic, not independent tests of model quality. Each row
repeats the same underlying question, and the three APIs encode disabled
reasoning differently. Streaming itself did not define a stable failure
boundary.

### Low-reasoning and replica controls

The same reproducer with low reasoning scored 600/600 at temperatures 0 and 1
for each checkpoint: **2,400/2,400 correct** in total. This control supports the
claim that the observed reproducer is specific to disabled reasoning under the
tested runtime. It does not prove that low reasoning makes general arithmetic
infallible.

An initial error concentration on one published-NVFP4 replica was not stable.
Two published-NVFP4 replicas and one QAD replica were each retested with 600
attempts at temperature 0/concurrency 1, temperature 0/concurrency 32, and
temperature 1/concurrency 32. All nine cells passed, **5,400/5,400**. The
evidence therefore does not identify a permanently defective replica, GPU, or
concurrency level. The captured runtime explicitly disabled deterministic
dynamic output, so temperature zero did not guarantee repeatable server output.

## Generalized direct-answer suite

The non-overlapping generalization suite contains 600 unique tasks, with 100 tasks
in each family:

- multiplication by a number immediately below a power of ten;
- multiplication selected to require several carry operations;
- addition with multi-digit carry chains;
- nonnegative subtraction with multi-digit borrow chains;
- exact integer division constructed from a divisor and integer quotient;
- a parenthesized multiplication followed by addition or subtraction.

The source `99 × 17` prompt is excluded. A fixed seed generates every operand
and Python integer arithmetic generates every answer key; no language model
wrote or judged the tasks. The frozen
[suite manifest](validation/direct-arithmetic-stability-suite-20260915.json)
has task-set SHA-256
`052618c42375ba8c2f73fe86ac280fc9256ea5921e69ca8609206230ba23ae80`.

Each request used OpenAI-compatible Chat Completions without streaming, one
user message, no system message, a 2,048-token output ceiling, no request seed,
and no retry. A response passed only when stripping leading and trailing
whitespace produced the exact expected base-10 integer. The test used three
attempts per task/checkpoint/temperature with reasoning disabled and one with
low reasoning. Temperatures 0 and 1 produced 9,600 requests across the two
checkpoints. All completed with HTTP and protocol success.

### Aggregate results

| Reasoning | Temperature | Published NVFP4 | QAD | QAD difference | Task-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| None | 0 | 1,424/1,800 (79.11%) | 1,455/1,800 (80.83%) | +1.72 points | +0.61 to +2.89 |
| None | 1 | 1,354/1,800 (75.22%) | 1,378/1,800 (76.56%) | +1.33 points | -0.33 to +3.00 |
| **None** | **0 and 1** | **2,778/3,600 (77.17%)** | **2,833/3,600 (78.69%)** | **+1.53 points** | **+0.50 to +2.56** |
| Low | 0 | 599/600 (99.83%) | 600/600 (100.00%) | +0.17 points | 0.00 to +0.50 |
| Low | 1 | 596/600 (99.33%) | 596/600 (99.33%) | 0.00 points | -0.83 to +0.83 |
| **Low** | **0 and 1** | **1,195/1,200 (99.58%)** | **1,196/1,200 (99.67%)** | **+0.08 points** | **-0.33 to +0.50** |

The 200,000-draw percentile bootstrap resamples the 600 task identifiers. All
temperatures and repetitions for a selected task remain inside its cluster.
Matching by task controls task difficulty, but outputs from the two checkpoints
were independently sampled and are not paired causal observations. The
generalization suite was designed after observing the source reproducer; its
intervals are evidence for these generated task families, not a preregistered
universal arithmetic claim.

### Results by operation family

The family rows combine temperatures 0 and 1 with reasoning disabled. They are
exploratory and have no multiple-comparison correction.

| Task family | Published NVFP4 | QAD | QAD difference | Task-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Borrow-chain subtraction | 582/600 (97.00%) | 580/600 (96.67%) | -0.33 points | -1.33 to +0.50 |
| Carry-chain addition | 500/600 (83.33%) | 526/600 (87.67%) | +4.33 points | +1.33 to +7.50 |
| Carry-heavy multiplication | 575/600 (95.83%) | 573/600 (95.50%) | -0.33 points | -2.17 to +1.50 |
| Exact integer division | 489/600 (81.50%) | 512/600 (85.33%) | +3.83 points | 0.00 to +7.67 |
| Near-power-of-ten multiplication | 583/600 (97.17%) | 584/600 (97.33%) | +0.17 points | -1.50 to +2.00 |
| Multiply then add/subtract | 49/600 (8.17%) | 58/600 (9.67%) | +1.50 points | -0.83 to +4.00 |

The largest observed QAD gains were carry-chain addition and exact division.
Only the carry-addition family has a strictly positive displayed interval;
the exact-division lower endpoint touches zero. Both checkpoints remained poor
on two-operation expressions, so QAD did not remove failures caused by asking
the no-reasoning path to perform a short sequence of arithmetic operations.

### Error shape

| Response classification with reasoning disabled | Published NVFP4 | QAD |
|---|---:|---:|
| Exact integer | 2,778 | 2,833 |
| Wrong integer, same digit count | 583 | 581 |
| Wrong integer, fewer digits | 141 | 109 |
| Wrong integer, more digits | 24 | 29 |
| Non-integer formatting or explanation | 74 | 48 |

QAD produced 55 more exact responses. The observed error-count reduction was
concentrated in shorter wrong integers (141 to 109) and format violations (74
to 48); same-length wrong integers barely changed. These counts describe
independently sampled outputs and do not prove that QAD transforms one specific
error into another.

Across both disabled-reasoning temperatures, published NVFP4 answered 370/600
tasks correctly in all six attempts and QAD answered 394/600 correctly in all
six. Conversely, 86 published-NVFP4 tasks and 89 QAD tasks were wrong in all
six attempts. QAD increased the fully stable set but did not shrink the set of
consistently failed tasks in this sample.

## What the evidence supports

The term *eliminates* is too strong for QAD under these conditions:

- QAD reduced the source reproducer's errors by 92%, but still failed twice.
- QAD improved the 600-task disabled-reasoning score by 1.53 points, with the
  largest observed gains in carry-chain addition and exact division.
- No generalized error class fell to zero. Same-length numeric errors remained
  almost unchanged, and both checkpoints failed over 90% of two-operation
  attempts.
- Low reasoning raised both checkpoints from approximately 77–79% to
  approximately 99.6%. For user-visible exact arithmetic, low reasoning or a
  calculator/verifier is a stronger mitigation than selecting QAD alone.

This comparison does not establish that quantization caused the source
failure. Both checkpoints are quantized and no BF16 checkpoint is present. No
SGLang arm was run, so the evidence also makes no cross-engine causal claim.
MTP3, FP8 KV cache, sampling, parser behavior, and serving kernels remain part
of the evaluated system.

## Checkpoints and runtime

The published checkpoint is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4@ada4da32a583a78aa47299f45a70603c950490b8`.
Its weight-index SHA-256 is
`435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9`.

The QAD checkpoint is identified by weight-index SHA-256
`c528e5628a0f448edc52023933c207a8dbded850afdd960d893c9171a7665dde`
and export-manifest SHA-256
`8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397`.
Its semantic training identity is 2,500 routed-expert trunk updates followed by
1,500 joint-refinement updates.

Both checkpoint pools used the same immutable image digest
`localinferencelab/vllm@sha256:23ab683d7ce32083f33c163df7dfe554b59aba75bbae8f5b8e4b5a2ef590209b`,
vLLM `0.26.1rc0+glm53.r36.vllm202a11a9`, one RTX PRO 6000 Blackwell GPU per
replica, BF16 activations, FP8 KV cache, three-token Multi-Token Prediction
(MTP3), prefix caching, 16 active sequences, and 517,581 logical KV-cache
tokens per replica. Published NVFP4 used GPUs 0–7 and QAD used GPUs 8–15 on the
same host. Each pool contained eight replicas and allowed four evaluation
requests per replica.

The public runtime manifests are the
[published-NVFP4 manifest](validation/aa-lcr-nvfp4-split-pool-runtime-manifest-20260915.json)
and [QAD manifest](validation/aa-lcr-qad-split-pool-runtime-manifest-20260915.json).
Their SHA-256 values are
`7db42f744820c663194d2d3ac39e6ced688daadce692066cc71f0017ee629fe8`
and `85cb9d4d3ed68995bb3f6c3e43bf92e6ae961635ade5f7bc3cbfbba9511eecda`.

## Reproduction and evidence

The generalized suite runner is
[`run-direct-arithmetic-stability.py`](tools/run-direct-arithmetic-stability.py),
SHA-256
`cfc5843aded9e2461b5a775fd6a1f3bf05eb0edbf08a69909c791bbdbf600351`.
It uses only Python's standard library and writes a resumable JSON Lines
journal. The analysis program is
[`analyze-direct-arithmetic-stability.py`](tools/analyze-direct-arithmetic-stability.py),
SHA-256
`6a071803498e3a7e2caac633039a958f3796da3d85568b9aacbe848702f8bd04`;
it requires NumPy for task-cluster bootstrap sampling.

```bash
export OPENAI_API_KEY='server-key-if-required'

python3 run-direct-arithmetic-stability.py \
  --endpoint-set NVFP4=http://HOST:30001/v1,http://HOST:30002/v1,http://HOST:30003/v1,http://HOST:30004/v1,http://HOST:30005/v1,http://HOST:30006/v1,http://HOST:30007/v1,http://HOST:30008/v1 \
  --endpoint-set QAD=http://HOST:30009/v1,http://HOST:30010/v1,http://HOST:30011/v1,http://HOST:30012/v1,http://HOST:30013/v1,http://HOST:30014/v1,http://HOST:30015/v1,http://HOST:30016/v1 \
  --model Qwen3.8-Flash-Next \
  --output-dir qwen38-direct-arithmetic-evidence \
  --tasks-per-family 100 \
  --none-repeats 3 \
  --low-repeats 1 \
  --temperatures 0 1 \
  --per-endpoint-concurrency 4
```

The public evidence files are:

- [machine-readable analysis](validation/direct-arithmetic-stability-analysis-20260915.json),
  SHA-256
  `d98ebb4f16ec7c2f3c95f02e1fb2e145d79a26f10579585cdc96e96c82125917`;
- [run configuration](validation/direct-arithmetic-stability-run-config-20260915.json),
  SHA-256
  `aa3827844d58af6045f6f418e4c01b3553b1ce70b31b415f3ca8875c798f2e8c`;
- [suite and answer key](validation/direct-arithmetic-stability-suite-20260915.json),
  SHA-256
  `ecffaa3d6d19e75dc49e515b1cc84ec78db0ed8685ead4dc39d149817900fbc5`;
- [runner summary](validation/direct-arithmetic-stability-summary-20260915.json),
  SHA-256
  `f8a43d381a053de0966d97a71394b228910b6ccc815ac18d7ccda4d81b6daf18`;
- [compressed per-attempt receipts](validation/direct-arithmetic-stability-attempts-20260915.jsonl.gz),
  compressed SHA-256
  `566074fc319cde531a58206e7b709d5804980567ba86bd2f2fee2122fcf22267`.

The decompressed attempt journal is 8,396,958 bytes with SHA-256
`a0bb63bc8309d567419e61e51d908acc121bfca95af41ab7f3280d9f2e29059b`.
The source-reproducer aggregates reference 112 disabled-reasoning receipts,
32 low-reasoning receipts, and nine replica-isolation receipts. Their canonical
receipt-set hashes are recorded in the machine-readable analysis. Raw source
reproducer receipts are retained by Local Inference Lab under `/mnt/luke/evals`.

## Attribution

The source single-prompt reproducer and `arithmetic_probe.py` were created by
**ktsaou**.

The generalized suite, evaluation execution, receipt validation, statistical
analysis, and report were created by **Martin Vit**, Local Inference Lab.
