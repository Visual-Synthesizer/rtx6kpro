# GLM-5.3-Flash QAD step 2,500 verifier-backed behavioral fidelity

Status: **qualified** for the declared 9,856-task primary comparison;
**research-only** as evidence about general model quality.

This report compares the published Local Inference Lab
GLM-5.3-Flash NVIDIA 4-bit floating-point (NVFP4) checkpoint with the
Quantization-Aware Distillation (QAD) step-2,500 checkpoint. The primary
analysis pools two non-overlapping Verifier-Backed Behavioral Fidelity (VBF)
task sets containing 7,168 and 2,688 paired tasks. Every task has an executable
answer key, every checkpoint completed every task without an API error, and
the serving configuration is matched between checkpoints within each task
set.

VBF's primary semantic score is the mean fraction of correct required fields
per task. Each component's suite and ±1-point practical-equivalence margin
were fixed before that component began. The equal-task pooling contract was
fixed before either QAD candidate execution began.

## Qualified result

| Published NVFP4 | QAD step 2,500 | QAD minus published NVFP4 | Paired 95% interval | Decision |
|---:|---:|---:|---:|---|
| 91.001% | 91.333% | **+0.332 points** | **-0.317 to +0.982 points** | **`practically_equivalent`** |

The complete 100,000-sample paired whole-task bootstrap interval lies inside
the declared ±1-point equivalence band. The result therefore establishes
practical equivalence on the primary VBF semantic score under this evaluation
contract.

The interval crosses zero, so it does not establish that QAD step 2,500 is
better. Its upper endpoint is only 0.018 percentage points inside the
equivalence boundary; the decision applies to the specified estimator,
bootstrap, suite allocation, and one-point margin rather than every reasonable
analysis or workload.

## Secondary outcome diagnostics

| Metric | Published NVFP4 | QAD step 2,500 | Difference |
|---|---:|---:|---:|
| Primary task-weighted semantic score | 91.001% | **91.333%** | **+0.332 points** |
| Completely correct tasks | **8,464/9,856 (85.877%)** | 8,328/9,856 (84.497%) | -1.380 points |
| Field micro-accuracy | **90,115/97,140 (92.768%)** | 90,057/97,140 (92.708%) | -0.060 points |
| Protocol-valid outputs | **9,518/9,856 (96.571%)** | 9,451/9,856 (95.891%) | -0.680 points |
| Length-limited outputs | **335/9,856 (3.399%)** | 404/9,856 (4.099%) | +69 outputs |

Published NVFP4 completed 1,029 tasks exactly when QAD did not; QAD completed
893 tasks exactly when published NVFP4 did not. The two-sided exact paired
McNemar p-value is 0.002067. Exact-task accuracy is a predeclared secondary
diagnostic, and its discordance favors published NVFP4 even though the primary
fractional score is practically equivalent.

At field-occurrence level, the comparison records 5,225
correct-to-incorrect regressions and 5,167 incorrect-to-correct recoveries,
for 58 net regressions across 11,977 value disagreements. Fields within a task
are dependent and are not treated as independent statistical samples. The
task-weighted semantic score and field-weighted micro-accuracy can move in
different directions because tasks contain different numbers of required
fields.

## Component comparisons

The pooled estimator gives every task equal weight. Hardware, host, topology,
and clock configuration are recorded as provenance but do not define
statistical strata.

| Execution component | Tasks | Published NVFP4 | QAD step 2,500 | Difference | Paired 95% interval | Decision |
|---|---:|---:|---:|---:|---:|---|
| Two TP4 replicas | 7,168 | 90.959% | 91.436% | +0.476 points | -0.276 to +1.226 | `not_worse` |
| One Max-Q TP4 replica | 2,688 | 91.111% | 91.058% | -0.053 points | -1.329 to +1.218 | `inconclusive` |
| Equal-task pooled estimate | 9,856 | 91.001% | 91.333% | **+0.332 points** | **-0.317 to +0.982** | **`practically_equivalent`** |

The 7,168-task component excludes degradation beyond one point but does not
fit entirely inside the equivalence band. The 2,688-task component alone is
less precise and remains inconclusive. Their point estimates have opposite
signs, but their intervals overlap substantially; neither component supports
a host- or GPU-specific behavioral claim.

## Task-family diagnostics

Each family contains 1,408 pooled task pairs. Family intervals are exploratory:
seven families were examined without a multiple-comparison correction, and no
family-specific deployment threshold was declared.

| Task family | Published NVFP4 | QAD step 2,500 | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constraint assignment | 99.564% | 99.409% | -0.154 points | -0.643 to +0.325 |
| Dependency graph | 96.347% | 95.992% | -0.355 points | -1.654 to +0.933 |
| Event-sourced state | **94.940%** | 90.936% | **-4.004 points** | **-5.673 to -2.344** |
| Evidence-chain retrieval | **92.229%** | 88.932% | **-3.297 points** | **-5.374 to -1.249** |
| Policy application | 99.821% | 99.808% | -0.013 points | -0.198 to +0.124 |
| Program execution | 65.138% | **74.513%** | **+9.375 points** | **+6.544 to +12.206** |
| Record reconciliation | 88.965% | 89.737% | +0.772 points | -1.039 to +2.583 |

QAD step 2,500 redistributes behavior rather than moving every family in one
direction. Its program-execution gain offsets lower event-sourced-state and
evidence-chain-retrieval scores in the aggregate. The family rows identify
replication and training targets; they do not convert the aggregate
equivalence result into three independent domain claims.

## What the result says about KLD training

The [QAD step-2,500 distribution-fidelity report](../../kld/glm-5.3-flash-qad-step2500.md)
shows that QAD step 2,500 reduces held-out natural-route Kullback-Leibler
Divergence (KLD) by 20.373% and exact-BF16-route KLD by 5.054% relative to
published NVFP4. Both predeclared KLD intervals exclude zero.

The VBF result establishes a different fact: that large distribution-fidelity
gain preserves the aggregate task-weighted semantic score within ±1 point on
9,856 deterministic tasks. It does not demonstrate an aggregate behavioral
improvement, and the exact-task and family diagnostics are not uniformly
favorable. Lower KLD is therefore evidence that the student better matches the
teacher distribution under the KLD contract; it is not by itself a monotonic
measure of downstream answer quality.

The combined evidence supports tracking both objectives during QAD:

- held-out full-distribution KLD detects whether training improves teacher
  fidelity;
- exact-route and natural-route KLD distinguish expert-computation fidelity
  from the executed routing system;
- VBF detects objectively scored behavioral regressions and recoveries; and
- capability evaluations remain necessary for workloads not represented by
  generated VBF tasks.

## Evaluation design

Python generators create each VBF prompt and executable answer key. No
language model writes the prompts, computes the expected answers, or judges a
response. The scorer extracts strict JSON and does not repair malformed data,
coerce types, ignore list order, or infer missing values. The
[VBF method page](verifier-backed-behavioral-fidelity.md) specifies the task
families, scoring rules, pairing, and decision vocabulary.

The primary comparison combines these immutable suites:

| Semantic role | Tasks | Tasks per family | Generator seed | Suite-file SHA-256 | Canonical task-record SHA-256 |
|---|---:|---:|---|---|---|
| Main two-replica component | 7,168 | 1,024 | `glm53-nvfp4-qad2500-confirmatory-20260906` | `38fc5741abb98cc188a5ecd80d2c7b0f328e9615725317b54d8b32c663fc0b42` | `7c5fc36de45e8b388c98279c1666aba22e3daa8f8273a11d2378794913c76f43` |
| Independent Max-Q component | 2,688 | 384 | `glm53-nvfp4-qad2500-maxq-replication-20260906` | `970b9022a28346de8a6cdbdbba47299c559e4b3737df03474161ebc1b15b844f` | `1656c1f7ac6dda18b0806a0a0862a2e75e6246c0a235ee88575ab4fcd75635f9` |

The two suites have no duplicate task identifier or prompt and do not overlap
the 224-task variance-estimation suite. All 9,856 tasks were required to
finish for both checkpoints. The declared sample was not stopped or extended
after either independent suite produced a QAD-versus-reference contrast.

The primary contrast is QAD step 2,500 minus published NVFP4. Its interval is
a 100,000-sample paired whole-task percentile bootstrap. The `worse`,
`better`, `practically_equivalent`, `not_worse`, and `inconclusive` decisions
use the predeclared ±0.01 semantic-score margin.

## Serving contracts

Both execution components use one generation per task, temperature zero,
maximum reasoning effort, one user message, no system message, a 32,768-token
output limit, and deterministic task-to-replica assignment. They use the
pinned image
`voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`.

Every TP4 replica uses Decode Context Parallelism 1, 32 scheduled sequences,
a 4,096-token scheduler budget, a 65,536-token model limit, a 9-GiB-per-GPU
FP8 key/value cache, CUDA graph capture sizes 1, 2, 4, 8, 16, and 32, and no
Multi-Token Prediction or speculative decoding.

| Component | Physical execution | Client concurrency | Request-seed base |
|---|---|---:|---:|
| 7,168 tasks | Two TP4 replicas on GPUs 8–11 and 12–15; RTX PRO 6000 Blackwell Workstation Edition | 64 | 20260906 |
| 2,688 tasks | One TP4 replica on GPUs 4–7; RTX PRO 6000 Blackwell Max-Q Workstation Edition | 32 | 20260907 |

Within each component, image, command, environment, topology, scheduler,
cache settings, physical GPU assignment, and request contract match between
checkpoints after excluding checkpoint identity and
representation-specific loader arguments. The 7,168-task component's
normalized runtime-contract SHA-256 values are
`b668306cbb91bdfa1a41687ed4ee86c16f82097d6fa9c4692dc82179142dd5f6`
and
`f42339f800f4f4f3a5c37ada933059c18da548c18c06c80dce1360794b74b3f3`.

The Max-Q devices use NVIDIA memory-transfer-rate offset `6000`, changing the
driver-reported maximum memory clock from 14,001 to 17,001 MHz. The first 173
successful published-NVFP4 receipts in that component were generated before
the offset was applied. The offset affects execution speed rather than the
requested sampling contract, but this asymmetry remains a limitation in the
runtime provenance.

## Checkpoint identity

| Role | Durable identity |
|---|---|
| Published NVFP4 reference | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62`; model-index SHA-256 `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb` |
| QAD step-2,500 candidate | Checkpoint-manifest SHA-256 `d783ff38cacd712bd29f7f7f31129b8633928c4bb782acf2e75b336cb90a743d`; model-index SHA-256 `b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b`; materialization SHA-256 `962f7905587be9e4377b8bebbcd8e93b49c3c49f3bda6d818260da518dcd5e0a` |

## Variance-estimation execution

A separate 224-task matched TP4×2 execution supplied the effect-size and
variance inputs used to select the 7,168-task sample size. Its tasks are not
part of the qualified 9,856-task estimate.

| Checkpoint | Semantic score | Completely correct tasks |
|---|---:|---:|
| Published NVFP4 | 92.31% | 194/224 (86.61%) |
| QAD step 1,750 | 91.87% | 192/224 (85.71%) |
| QAD step 2,500 | 89.72% | 190/224 (84.82%) |

QAD step 2,500 minus published NVFP4 was -2.59 points with a paired 95%
interval from -6.93 to +1.67 points. QAD step 2,500 minus QAD step 1,750 was
-2.15 points with an interval from -6.44 to +2.11 points. Both decisions were
`inconclusive`. The 224-task negative estimate did not replicate on the two
independent suites, but its wide interval contains the qualified pooled
estimate and is not statistically contradictory.

The 224-task execution itself is qualified, but its QAD step-1,750 and
published-NVFP4 controls were commissioned after inspecting the step-2,500
answers. It remains variance-estimation and historical comparison evidence,
not part of the primary confirmatory result.

## Scope and limitations

The qualified claim concerns the task-weighted semantic score on seven
procedurally generated deterministic task families. It does not establish
equivalence for exact-task accuracy, individual families, free-form chat,
creative writing, factual recall, safety, tool use, or arbitrary deployment
traffic.

The pooled analysis deliberately uses no host, GPU, topology, or clock strata.
Both checkpoint arms are runtime-matched within each component, and the two
component estimates are reported separately so that their uncertainty remains
visible.

The family intervals and secondary diagnostics explain heterogeneity but were
not separately powered confirmatory endpoints. The exact-task McNemar result
is evidence of a stricter-output disadvantage for QAD step 2,500; it must not
be erased by the aggregate semantic-equivalence label.

## Reproducibility record

The [machine-readable public summary](validation/qad-step2500-vbf-pooled-9856-20260907.json)
contains full-precision aggregate values and component provenance.

| Evidence | Durable identifier |
|---|---|
| 7,168-task evaluation contract | SHA-256 `e4fd526b019f33ae4f3605bce13f0954ff6fb62d9511748b8d7f91a37c68eb62` |
| 2,688-task evaluation contract | SHA-256 `fa63d21d6e0da8ba3f7a86049c2bb60ec6079bc86ad5382b32324228001fb0b9` |
| 9,856-task pooled contract | SHA-256 `447d37915147e85fbe528cef9735aa2b9739b100d3775dfc62d083890037bf98` |
| 7,168-task runs | Published run `2026-09-06-a41f6b0f0a06a94b`, receipt SHA-256 `0bbeed63e3bdbfdc55bcf7f53f971ca1867fb6229d5f7d1930033d723f0ebbd4`; QAD run `2026-09-06-b4453f873b8b8d4f`, receipt SHA-256 `789835d32fea28ca19eab3b13010942fcc9b7418767b061b1e6aead979d459a6` |
| 2,688-task runs | Published run `2026-09-06-3f2017143326cb7d`, receipt SHA-256 `617245343c32abb998ff9bf1327750b681659a4325bf3fd171b59949f18129d8`; QAD run `2026-09-06-01ff83c7e3f494bb`, receipt SHA-256 `a39bf47453c1ad23ff8eb80b0657b0c03359f7a3c865ec6296d6f153f72df440` |
| 7,168-task comparison | JSON SHA-256 `b5e432117351791155799a7b5bd953a1e33b85516c7b90e79e9f7aae53f9ba82`; Markdown SHA-256 `93e465c83c4c71cc3f9c019e5cac5e882740a12d9f71af6e6b25cc28ccf0ed35` |
| 2,688-task comparison | JSON SHA-256 `4b288f7775feef8b1b60ff213c64710f7a54f10fd45cc5845a66179e8ec57ed3`; Markdown SHA-256 `a5911110d84f80fad602a875b597d623be477017b256846af7d808688e1e6f0a` |
| Pooled comparison | JSON SHA-256 `29dd86a73d4b0f1fb75b1e8598aeeb18b456940564e239bb346411040d916408`; Markdown SHA-256 `ff7a4daefc6ba42c5aac9cc94b49d88fd7e3402a46d9e8608ee2378ade715254` |
| Public machine-readable summary | SHA-256 `c792de2cfca751f185e5bb15e027c6cd7ef7bc31c50544c7f990345b48160251` |
| Evaluated implementation | `local-inference-lab/llm-inference-bench`, package `behavioral_fidelity`, Git commit `db4e5d9f28ad66f769cef2ef2365e9d24dd7969b`; run manifests also pin five implementation-file hashes |
| Retained 7,168-task artifact | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/nvfp4-qad-step2500-confirmatory-vbf-7168-20260906` |
| Retained 2,688-task artifact | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/nvfp4-qad-step2500-maxq-replication-vbf-2688-20260906` |
| Retained pooled report | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/nvfp4-qad-step2500-pooled-vbf-9856-20260906` |

## Attribution

The deterministic task generators, executable answer keys, strict scoring
contract, durable receipt format, runtime-verification records, component
executions, pooled analysis, and report are Local Inference Lab work. Paired
bootstrap intervals and McNemar's exact test are established statistical
methods rather than claimed mathematical inventions.
