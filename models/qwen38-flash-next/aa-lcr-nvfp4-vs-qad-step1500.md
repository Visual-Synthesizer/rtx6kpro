# Qwen3.8-Flash-Next AA-LCR: published NVFP4 versus QAD step 1,500

Status: **qualified local reproduction**.

This report compares the published NVIDIA 4-bit floating-point (NVFP4)
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` checkpoint at revision
`ada4da32a583a78aa47299f45a70603c950490b8` with the Local Inference Lab
quantization-aware-distillation checkpoint
`Qwen3.8-Flash-Next-NVFP4-QAD-step1500-v1`. Each checkpoint produced ten
independent answers for every one of the 100 Artificial Analysis Long Context
Reasoning (AA-LCR) v1.1 questions. GPT-5.6 Luna at medium reasoning classified
all 2,000 answers with the AA-LCR v1.1 equality-checker contract.

The QAD checkpoint scored **794/1,000 = 79.4%**, compared with
**775/1,000 = 77.5%** for published NVFP4, a point-estimate difference of
**+1.9 percentage points**. The exact question-cluster bootstrap 95% percentile
interval is **0.0 to +3.8 percentage points**. Its lower endpoint touches zero,
so the result favors QAD but does not provide strict positive separation under
that two-sided interval criterion. The measured magnitude remains uncertain.

This ten-generation evaluation extends the official three-generation AA-LCR
aggregation. It is not an official Artificial Analysis leaderboard result.

Tensor parallelism of one (TP1) assigns one GPU to each replica. Multi-token
prediction with three speculative tokens is abbreviated MTP3.

## Result

| Checkpoint and serving configuration | Correct | AA-LCR pass@1 reproduction |
|---|---:|---:|
| Published NVFP4, TP1, MTP3 | 775/1,000 | **77.5%** |
| QAD step 1,500 v1, TP1, MTP3 | 794/1,000 | **79.4%** |
| QAD minus published NVFP4 | +19/1,000 | **+1.9 percentage points** |

| Global repeat | Published NVFP4 | QAD step 1,500 | Difference |
|---:|---:|---:|---:|
| 0 | 77/100 | 79/100 | +2 points |
| 1 | 75/100 | 80/100 | +5 points |
| 2 | 79/100 | 81/100 | +2 points |
| 3 | 75/100 | 77/100 | +2 points |
| 4 | 80/100 | 79/100 | -1 point |
| 5 | 79/100 | 78/100 | -1 point |
| 6 | 80/100 | 80/100 | 0 points |
| 7 | 76/100 | 81/100 | +5 points |
| 8 | 79/100 | 83/100 | +4 points |
| 9 | 75/100 | 76/100 | +1 point |

QAD scored higher in seven repeat aggregates, published NVFP4 scored higher in
two, and one tied. At the question level, QAD had a higher ten-generation mean
on 24 questions, published NVFP4 had a higher mean on 15, and 61 tied.

The three-generation segment alone measured 77.0% versus 80.0%, a QAD
difference of +3.0 points. The independent seven-generation segment measured
77.71% versus 79.14%, a difference of +1.43 points. The combined estimate is
therefore smaller than the three-generation estimate; increasing the generation
count materially changed the estimated magnitude.

## Statistical interpretation

The comparison receipt computes the empirical bootstrap distribution exactly,
without Monte Carlo draws. It resamples the 100 question clusters with
replacement and retains all ten stochastic generations from both checkpoints
inside every selected cluster. The resulting intervals are:

| Quantity | 95% percentile interval |
|---|---:|
| Published NVFP4 score | 70.8% to 83.7% |
| QAD step 1,500 score | 72.7% to 85.6% |
| QAD minus published NVFP4 | **0.0 to +3.8 points** |

The bootstrap distribution assigns 2.697% probability to a resampled
difference at or below zero. That bootstrap tail fraction is descriptive and
is not presented as a preregistered hypothesis-test p-value. Matching repeat
numbers also do not pair the independently sampled candidate answers, so the
attempt-level label table does not define a McNemar test.

The supported conclusion is that this sample favors QAD on AA-LCR v1.1. It
does not establish a precise gain, universal capability improvement, or strict
positive separation under the declared two-sided interval rule. Additional
questions would address task-sampling uncertainty more directly than further
repeats of the same 100 questions; repeated equality-checker labels would
quantify judge variation.

The [machine-readable comparison](validation/aa-lcr-nvfp4-vs-qad-step1500-ten-generations-20260915.json)
has SHA-256
`915c4d8a4ff3bb08488f9a6d357d345eaa06850e8f4d9b09fa8fbf11a8bc57bf`.

## Category observations

Category differences are descriptive. Several categories contain very few
question clusters, so their repeated answers are not independent substitutes
for additional questions.

| Document category | Questions | Published NVFP4 | QAD step 1,500 | Difference |
|---|---:|---:|---:|---:|
| Academia | 5 | 44/50 (88.0%) | 47/50 (94.0%) | +6.0 points |
| Company documents | 63 | 500/630 (79.37%) | 523/630 (83.02%) | +3.65 points |
| Government consultations | 11 | 72/110 (65.45%) | 66/110 (60.0%) | -5.45 points |
| Industry reports | 8 | 65/80 (81.25%) | 62/80 (77.5%) | -3.75 points |
| Legal | 6 | 40/60 (66.67%) | 41/60 (68.33%) | +1.67 points |
| Marketing | 6 | 44/60 (73.33%) | 46/60 (76.67%) | +3.33 points |
| Survey reports | 1 | 10/10 (100.0%) | 9/10 (90.0%) | -10.0 points |

These cells do not establish domain-specific gains or regressions. In
particular, ten responses to one survey question remain one question cluster.

## Dataset and generation contract

| Property | Qualified value |
|---|---|
| Dataset | `ArtificialAnalysis/AA-LCR` |
| Dataset revision | `9a77ef56b717057ade24ceab4d273712a0b4f19e` |
| Question CSV SHA-256 | `aea5198982436fa774f616964bbeb34716de9e94368a123275b9355765c9b73b` |
| Extracted-document ZIP SHA-256 | `5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a` |
| Questions | 100 |
| Independent generations per checkpoint and question | 10 |
| Prompt messages | one user message; no system message |
| Candidate reasoning effort | `xhigh` |
| Temperature / top-p / top-k / min-p | `1.0` / `0.95` / `20` / `0` |
| Presence / repetition penalty | `0` / `1` |
| Request seed | omitted |
| Maximum output | 139,000 tokens |
| Streaming | disabled |
| Completion status | all 2,000 responses ended with `stop` |

Published NVFP4 generated 2,039,403 completion tokens across its ten
generations; QAD generated 2,164,009. Both checkpoints processed 106,617,560
prompt tokens. The 139,000-token value is a ceiling rather than a requested
response length.

Published NVFP4 returned an empty final-answer field twice: question 55 in
global repeat 0 and question 54 in global repeat 8. Both requests ended with
`stop`, retained non-empty reasoning traces, and were evaluated once as empty
candidate answers. Both received `INCORRECT`; neither was regenerated. QAD
returned no empty final answers.

## Checkpoints and serving runtime

The published checkpoint identity is
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4@ada4da32a583a78aa47299f45a70603c950490b8`.
Its weight-index SHA-256 is
`435eef76fc10fc6e932a208a1f85a86bc9c7ffc389b27b34ba83bb6a5d0371e9`.

The QAD checkpoint is the local artifact
`/data/models/Qwen3.8-Flash-Next-NVFP4-QAD-step1500-v1`, identified by
weight-index SHA-256
`c528e5628a0f448edc52023933c207a8dbded850afdd960d893c9171a7665dde`
and export-manifest SHA-256
`8a0b93599e3edb4ab25357e8af16cf1ac2c6a61354fcec9d6aa50ee9fbf94397`.
Its semantic training identity is 2,500 routed-expert trunk updates followed by
1,500 joint-refinement updates.

Every replica used one NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU,
the same immutable container digest
`localinferencelab/vllm@sha256:23ab683d7ce32083f33c163df7dfe554b59aba75bbae8f5b8e4b5a2ef590209b`,
bfloat16 (BF16) activations, 8-bit floating-point (FP8) key/value (KV) cache,
MTP with three speculative tokens, prefix caching, 262,144-token maximum
context, 16 active sequences, and 517,581 logical KV-cache tokens. The
evaluated artifact is the complete served system; the result does not isolate
weights from cache quantization, kernels, or speculative decoding.

The evaluation consists of two immutable generation segments:

- Global repeats 0–2 used 16 TP1 replicas for one checkpoint at a time, with
  256 total client workers.
- Global repeats 3–9 used simultaneous, disjoint pools on the same Frank2 host:
  published NVFP4 on GPUs 0–7 and ports 30001–30008, and QAD on GPUs 8–15 and
  ports 30009–30016. Each pool used 128 total client workers.

The per-replica serving contract and stochastic sampling parameters are stable
within each checkpoint across both segments. Global pool size and scheduler
history differ between the two segments, so the report qualifies the declared
checkpoint-and-serving configurations rather than checkpoint weights alone.

The public runtime receipts are:

- [published NVFP4 sequential-pool runtime](validation/aa-lcr-nvfp4-sequential-pool-runtime-manifest-20260915.json),
  SHA-256
  `2516cf648003a6e2dddb6afa4b7872b362810eddb53b95edde21ff9349fb5635`;
- [QAD sequential-pool runtime](validation/aa-lcr-qad-step1500-sequential-pool-runtime-manifest-20260915.json),
  SHA-256
  `ef73fff3e634b1b65b96d438c22e20e22814c0cd44c0ef4a8a0f584e619a8439`;
- [published NVFP4 split-pool runtime](validation/aa-lcr-nvfp4-split-pool-runtime-manifest-20260915.json),
  SHA-256
  `7db42f744820c663194d2d3ac39e6ced688daadce692066cc71f0017ee629fe8`;
- [QAD split-pool runtime](validation/aa-lcr-qad-step1500-split-pool-runtime-manifest-20260915.json),
  SHA-256
  `85cb9d4d3ed68995bb3f6c3e43bf92e6ae961635ade5f7bc3cbfbba9511eecda`.

## Equality checker

GPT-5.6 Luna with medium reasoning evaluated each answer once. Every attempt
used a fresh Codex CLI 0.154.0 session in an empty read-only workspace with user
configuration disabled. The checker received the question, official answer,
and candidate answer, but not the candidate checkpoint identity. It emitted a
single `CORRECT` or `INCORRECT` JSON verdict.

All 2,000 expected verdict receipts are present, qualified against the same
dataset revision and equality-checker prompt SHA-256
`e5ed5b4646e01151f43e9ea97ba8490208151d4317c594963719f37dcecf40ca`.
No generation or judge failure sidecar exists.

Artificial Analysis specifies GPT-5.6 Luna at medium reasoning for AA-LCR v1.1.
The source is the
[Artificial Analysis Intelligence Benchmarking Methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking).

## Qualification receipts

Each linked generation manifest defines the sampling and runtime contract. Each
completeness receipt verifies all expected responses and hashes. Each judge
summary verifies all expected verdicts and identifies its retained receipt set.

| Segment | Generation contract | Completeness | Luna summary |
|---|---|---|---|
| Published NVFP4, repeats 0–2 | [manifest](validation/aa-lcr-nvfp4-sequential-pool-generation-manifest-20260915.json) | [receipt](validation/aa-lcr-nvfp4-sequential-pool-generation-completeness-20260915.json) | [summary](validation/aa-lcr-nvfp4-sequential-pool-luna-summary-20260915.json) |
| Published NVFP4, repeats 3–9 | [manifest](validation/aa-lcr-nvfp4-split-pool-generation-manifest-20260915.json) | [receipt](validation/aa-lcr-nvfp4-split-pool-generation-completeness-20260915.json) | [summary](validation/aa-lcr-nvfp4-split-pool-luna-summary-20260915.json) |
| QAD step 1,500, repeats 0–2 | [manifest](validation/aa-lcr-qad-step1500-sequential-pool-generation-manifest-20260915.json) | [receipt](validation/aa-lcr-qad-step1500-sequential-pool-generation-completeness-20260915.json) | [summary](validation/aa-lcr-qad-step1500-sequential-pool-luna-summary-20260915.json) |
| QAD step 1,500, repeats 3–9 | [manifest](validation/aa-lcr-qad-step1500-split-pool-generation-manifest-20260915.json) | [receipt](validation/aa-lcr-qad-step1500-split-pool-generation-completeness-20260915.json) | [summary](validation/aa-lcr-qad-step1500-split-pool-luna-summary-20260915.json) |

The aggregate comparison records all segment hashes and candidate and judge
receipt-set identities. Candidate responses, complete API response objects,
per-attempt hashes, and judge verdicts are retained by Local Inference Lab under
`/mnt/luke/evals`; they are not distributed in this repository.

The retained generation runner has SHA-256
`97c5be57d01c4fcd19ba18f75da03f582d94a7062f18423042119d7acd8351e1`.
The equality-checker runner has SHA-256
`56960fbc0a46916090b75e66494a54104a75e15e5643e3c6354d2c379b562934`.
The segmented comparison runner has SHA-256
`c23792c333ef8eeab1fdf49a407e8b2cdd665c4abc5382d8b30bf0184a5d1430`.

## Interpretation limits

- The 100 questions do not represent every long-context user workload.
- Ten stochastic generations cannot eliminate candidate-generation variance.
- Each answer received one equality-checker verdict, so judge repeatability is
  not measured by this artifact.
- Public AA-LCR questions and answers create contamination risk.
- Category rows contain between one and 63 questions and are not separately
  powered tests.
- No throughput comparison follows from wall time because global serving pool
  sizes and scheduler histories differ.

The QAD checkpoint is **research-only** as a deployment target. The AA-LCR
artifact is **qualified** for the exact checkpoint, runtime, dataset, sampling,
and equality-checker identities reported above.
