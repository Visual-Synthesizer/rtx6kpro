# GLM-5.3-Flash QAD step 2,500 verifier-backed behavioral fidelity

Status: **qualified executions; inconclusive matched-profile comparisons;
research-only behavioral claim**.

This report compares the published Local Inference Lab
GLM-5.3-Flash NVIDIA 4-bit floating-point (NVFP4) checkpoint with two
Quantization-Aware Distillation (QAD) materializations at training steps 1,750
and 2,500. Every checkpoint answered the same 224 deterministic
Verifier-Backed Behavioral Fidelity (VBF) tasks under the same two-replica
Tensor Parallelism 4 serving profile.

VBF computes prompts and answer keys with Python generators and scores strict
JSON without a language-model judge. The primary semantic score is the mean
fraction of correct required fields per task. Exact-task accuracy requires the
complete returned object to match its answer key.

## Result

| Checkpoint | Semantic score | Exact tasks | Field micro-accuracy | Protocol valid | Length limited |
|---|---:|---:|---:|---:|---:|
| Published NVFP4 | **92.31%** | **194/224 (86.61%)** | **93.99%** | 215/224 (95.98%) | 7 |
| QAD step 1,750 | 91.87% | 192/224 (85.71%) | 93.72% | **219/224 (97.77%)** | **5** |
| QAD step 2,500 | 89.72% | 190/224 (84.82%) | 90.76% | 213/224 (95.09%) | 11 |

QAD step 2,500 has the lowest point estimate on the primary metric, exact-task
accuracy, field micro-accuracy, and output completion. The paired uncertainty
does not resolve the aggregate differences, however. The supported conclusion
is not that QAD step 2,500 is worse; it is that this 224-task suite does not
demonstrate improvement, non-inferiority, equivalence, or degradation inside
the declared one-percentage-point margin.

## Paired comparisons

Each comparison subtracts the named reference's score from the candidate's
score on every task. Confidence intervals use 100,000 paired task-cluster
bootstrap samples. “Exact harm” means that the reference completed the task
exactly and the candidate did not; “recovery” means the reverse.

| Candidate minus reference | Semantic difference | Paired 95% interval | One-point decision | Exact harm / recovery | McNemar p |
|---|---:|---:|---|---:|---:|
| QAD step 2,500 minus published NVFP4 | **-2.59 points** | **-6.93 to +1.67** | `inconclusive` | 24 / 20 | 0.6516 |
| QAD step 2,500 minus QAD step 1,750 | **-2.15 points** | **-6.44 to +2.11** | `inconclusive` | 21 / 19 | 0.8746 |
| QAD step 1,750 minus published NVFP4 | -0.44 points | -4.55 to +3.68 | `inconclusive` | 23 / 21 | 0.8804 |

The QAD step-2,500 comparison with published NVFP4 contains 161
correct-to-incorrect field changes and 90 recoveries across 53 tasks with a
changed value. Its comparison with QAD step 1,750 contains 153 field
regressions and 88 recoveries across 50 changed tasks. These counts describe
where answers moved; fields within one task are not independent statistical
samples.

## Task-family diagnostics

The family rows are exploratory. Seven families were examined without a
multiple-comparison correction, and the matched control executions were
commissioned after the QAD step-2,500 result was inspected.

| Task family | Published NVFP4 | QAD step 1,750 | QAD step 2,500 | Step 2,500 minus published NVFP4 | Step 2,500 minus step 1,750 |
|---|---:|---:|---:|---:|---:|
| Constraint assignment | 100.00% | 100.00% | 100.00% | +0.00 (+0.00 to +0.00) | +0.00 (+0.00 to +0.00) |
| Dependency graph | 92.86% | 93.75% | 95.98% | +3.12 (-1.79 to +10.27) | +2.23 (-7.14 to +12.05) |
| Event-sourced state | 94.14% | 94.92% | 87.11% | -7.03 (-17.97 to +1.56) | -7.81 (-21.48 to +5.08) |
| Evidence-chain retrieval | 100.00% | 95.83% | 83.85% | -16.15 (-29.17 to -4.17) | -11.98 (-26.56 to +1.04) |
| Policy application | 99.16% | 99.75% | 98.77% | -0.39 (-3.22 to +1.89) | -0.98 (-2.93 to +0.00) |
| Program execution | 70.98% | 61.61% | 70.54% | -0.45 (-20.98 to +20.09) | +8.93 (-9.38 to +27.23) |
| Record reconciliation | 89.06% | 97.27% | 91.80% | +2.73 (-9.38 to +16.02) | -5.47 (-14.45 to +1.17) |

QAD step 2,500 has a lower evidence-chain-retrieval score than published
NVFP4, and the paired family-level 95% interval is entirely below zero. Five
of its eleven length-limited answers belong to that family, compared with none
for published NVFP4. This is a concrete replication target, not a qualified
domain-level claim: the family analysis is exploratory, the suite was
previously used, and the aggregate comparison remains inconclusive.

QAD step 2,500 improves the program-execution point estimate relative to QAD
step 1,750 while reducing the event-sourced-state, evidence-chain-retrieval,
and record-reconciliation estimates. The opposing changes explain why a
single aggregate score is not a sufficient checkpoint diagnosis.

## Matched execution profile

All three checkpoints used:

- two serving replicas, each with Tensor Parallelism 4 and Decode Context
  Parallelism 1;
- physical GPUs 8–11 for endpoint port 5054 and GPUs 12–15 for endpoint port
  5055;
- 32 concurrent sequences per replica and 64 concurrent client requests;
- a 4,096-token scheduler budget, 65,536-token model context, and 32,768-token
  output limit;
- FP8 key/value cache with 9 GiB allocated per GPU;
- Multi-Token Prediction and speculative decoding disabled;
- CUDA graph capture sizes 1, 2, 4, 8, 16, and 32;
- one user message, no system message, temperature zero, maximum reasoning
  effort, one generation per task, and request-seed base 20260904; and
- container image
  `voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`.

The normalized checkpoint-runtime contract SHA-256 values are
`b668306cbb91bdfa1a41687ed4ee86c16f82097d6fa9c4692dc82179142dd5f6`
for the GPU 8–11 replica and
`f42339f800f4f4f3a5c37ada933059c18da548c18c06c80dce1360794b74b3f3`
for the GPU 12–15 replica. Each value is identical across published NVFP4,
QAD step 1,750, and QAD step 2,500. Ports, served names, checkpoint paths, and
representation-specific loader arguments are excluded from normalized
checkpoint identity.

The three complete task executions took 17.27 minutes for published NVFP4,
17.05 minutes for QAD step 1,750, and 19.68 minutes for QAD step 2,500. No API
request failed. The deterministic task-to-endpoint assignment sent 114 tasks
to the GPU 8–11 replica and 110 tasks to the GPU 12–15 replica for every
checkpoint.

## Scope and limitations

The 224-task suite was generated and used for the BF16, published-NVFP4, and
QAD step-1,750 TP8 report before the QAD step-2,500 execution. The suite gives
exact paired continuity but is not an independent held-out replication for
QAD step 2,500.

The QAD step-1,750 and published-NVFP4 TP4×2 controls were commissioned after
the QAD step-2,500 TP4×2 result was inspected. Their runtime matching removes
the known serving-profile difference, but the resulting comparisons remain
post-hoc evidence. A separately generated and frozen suite is required for a
confirmatory behavioral claim.

Tensor parallel layouts, request batching, and GPU kernels can change greedy
outputs through floating-point and scheduling nondeterminism even when they
serve the same checkpoint. This report compares only the matched TP4×2
executions. The separate [TP8 VBF report](verifier-backed-behavioral-fidelity.md)
remains the qualified record for BF16, published NVFP4, and QAD step 1,750
under its own matched profile; scores from the two profiles are not pooled.

VBF measures deterministic instruction execution on generated tasks with
complete facts in the prompt. It does not establish general chat quality,
creative-writing quality, factual recall, safety, or every deployment
workload.

## Relationship to distribution fidelity

The [QAD step-2,500 distribution-fidelity report](../../kld/glm-5.3-flash-qad-step2500.md)
finds that QAD step 2,500 reduces natural-route Kullback-Leibler Divergence
(KLD) by 20.37% relative to published NVFP4 and by 10.73% relative to QAD step
1,750 on the held-out KLD partition. Under exact BF16-route replay, it reduces
KLD by 5.05% relative to published NVFP4 but increases KLD by 5.50% relative
to QAD step 1,750.

The VBF point estimates do not follow the natural-route KLD ordering:
published NVFP4 scores highest, followed by QAD step 1,750 and QAD step 2,500.
The aggregate VBF intervals are inconclusive, so the combined evidence means
only that improved next-token distribution fidelity did not demonstrate a
behavioral improvement on this task suite. KLD and VBF measure different
estimands and neither result invalidates the other.

## Reproducibility record

| Evidence | Durable identifier |
|---|---|
| QAD step-2,500 checkpoint | Quatrain checkpoint-manifest SHA-256 `d783ff38cacd712bd29f7f7f31129b8633928c4bb782acf2e75b336cb90a743d`; materialization SHA-256 `962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a` |
| QAD step-1,750 checkpoint | Quatrain checkpoint-manifest SHA-256 `ed76ada3ee9e4bf10d74554fb62a8d0e1767e8d5b22ee15f4940cc10cbf2da5c`; materialization SHA-256 `4c8439a5e8891f0dc7f22134803e44aebd4296284e356ae0a8bb2cf76e4ee937` |
| Published NVFP4 checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |
| Suite | 224 tasks; suite-file SHA-256 `67a7674a731dc9cd51697abe9c2bce9f3d406c1b211c6f947de1861004ac8b47`; canonical task-record SHA-256 `e6d71089c599db57f77b894ab693186100d81d8d15ad35017e8fc6870eac859d` |
| Published NVFP4 run | Run ID `2026-09-06-0c6427421361b1fb`; runtime-bundle SHA-256 `2734bb334bd0deb8315df61d1d04282f5db6acd4432d2be1340f2fe42062f51c`; receipt-set SHA-256 `a142e5df5012305b93cd5c734c4b2377f2a4e9cf852f2cb5b549f729eb6e7e27` |
| QAD step-1,750 run | Run ID `2026-09-06-478b0b08fcc1ec7a`; runtime-bundle SHA-256 `b8b39d165cea82a4f1e43128de45b0783350ce3bc7407932ae21ed8f3e51566e`; receipt-set SHA-256 `d2dd4b7b31562c82653916a86ed7626cb5a940bae73921c4ab5baf7a8016bec0` |
| QAD step-2,500 run | Run ID `2026-09-06-5ee22d3598d40d00`; runtime-bundle SHA-256 `1163c791ea17721628b6259fd8a84a919147202b30ceb1c71066d052b6eeeb29`; receipt-set SHA-256 `e4ef4e184f047f16636781b7742adfda87467867cf87fe4df974854855e4a066` |
| Step 2,500 versus published comparison | JSON SHA-256 `deb0ad1b3569459260db58d606e9b5143365fbd544243dd560884e2e0553e1ca` |
| Step 2,500 versus step 1,750 comparison | JSON SHA-256 `53965f2b6b0baa8254b969513a6adffbf9b12d72bfbe2fe421ca990b6c911220` |
| Step 1,750 versus published comparison | JSON SHA-256 `34cf8820e492904c0d88dfa7a71f0ef476587ea078aa9d145e5a4bb7fcc78a36` |
| Post-hoc matched-runtime plan | SHA-256 `394979e777f8896248a63029089688825bacb83857ba3dd726aee211e5447445` |
| Evaluated implementation | `local-inference-lab/llm-inference-bench`, package `behavioral_fidelity`, Git commit `db4e5d9f28ad66f769cef2ef2365e9d24dd7969b` |
| Retained artifact root | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/qad-step2500-checkpoint-comparison-vbf-224-20260906` |

## Attribution

The deterministic task generators, strict scoring contract, durable receipts,
two-replica execution records, and paired comparison implementation are Local
Inference Lab work. Paired bootstrap intervals and McNemar's exact test are
established statistical methods rather than claimed mathematical inventions.
