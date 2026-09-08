# GLM-5.3-Flash QAD step 2,500 verifier-backed behavioral fidelity

Status: **qualified** for the declared 9,856-task aggregate comparison and
the separately declared 2,048-task program-execution comparison;
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

Each family contains 1,408 pooled task pairs. These rows are diagnostics from
the aggregate comparison: seven families were examined without a
multiple-comparison correction, and no family-specific deployment threshold
was declared. The program-execution row also has the separately frozen,
non-overlapping confirmation reported below; the other six rows remain
exploratory.

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
evidence-chain-retrieval scores in the aggregate. The aggregate equivalence
result does not establish equivalence within each family.

## Qualified program-execution confirmation

Status: **qualified** for unseen numeric instances of the deterministic
integer-program template; **research-only** for programming or tool-use
capability.

The program-execution comparison uses 2,048 additional task pairs whose prompt
content does not overlap the 1,408 pairs in the aggregate comparison. Its
primary endpoint and decision rule were fixed before either checkpoint
answered these tasks. Confirmation required all task pairs without request
errors, matched runtime evidence within each block, a positive point estimate
in both counterbalanced blocks, and a pooled 100,000-sample paired-bootstrap
95% interval entirely above a one-percentage-point practical-improvement
boundary.

| Task pairs | Published NVFP4 | QAD step 2,500 | QAD minus published NVFP4 | Paired 95% interval | Decision |
|---:|---:|---:|---:|---:|---|
| 2,048 | 64.927% | **73.298%** | **+8.371 points** | **+5.985 to +10.742 points** | **`confirmed_practical_gain`** |

The result satisfies every declared confirmation condition. It establishes a
practically meaningful QAD advantage for mental execution of unseen numeric
instances of the specified program template. It does not alter the
9,856-task aggregate `practically_equivalent` decision, because the aggregate
and program-specific comparisons answer different questions.

### Counterbalanced execution blocks

| Block | GPUs | Checkpoint order | Task pairs | Difference | Paired 95% interval |
|---|---|---|---:|---:|---:|
| A | 8–11 | Published NVFP4, then QAD step 2,500 | 1,024 | +7.338 points | +4.032 to +10.645 points |
| B | 12–15 | QAD step 2,500, then published NVFP4 | 1,024 | +9.403 points | +6.013 to +12.793 points |

The block order is reversed across the two GPU groups. Both block intervals
exclude zero, so the pooled result is not carried by one GPU group or one
checkpoint order.

### Exact-task and difficulty diagnostics

Published NVFP4 solved 1,190/2,048 tasks exactly (58.105%); QAD step 2,500
solved 1,381/2,048 exactly (67.432%). QAD alone solved 494 tasks, while the
reference alone solved 303. The two-sided exact paired McNemar p-value is
`1.36125e-11`. Exact-task accuracy is secondary, but it supports the same
direction as the primary fractional score.

| Difficulty | Tasks | Published NVFP4 | QAD step 2,500 | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Standard | 683 | 85.589% | 88.831% | +3.242 points | +0.063 to +6.442 points |
| Demanding | 683 | 64.568% | 71.784% | +7.216 points | +2.928 to +11.483 points |
| Stress | 682 | 44.596% | 59.258% | +14.663 points | +9.929 to +19.376 points |

The gain appears in all three generated difficulty strata and grows with loop
length. These strata are diagnostics rather than separately declared
hypotheses.

### Required-field diagnostics

| Required field | Published NVFP4 | QAD step 2,500 | Difference |
|---|---:|---:|---:|
| Final register `a` | 67.627% | 75.684% | +8.057 points |
| Final register `b` | 67.432% | 75.488% | +8.057 points |
| Final register `c` | 68.506% | 76.367% | +7.861 points |
| Final register `d` | 68.701% | 76.562% | +7.861 points |
| Emitted sequence | 59.766% | 68.848% | +9.082 points |
| Odd-position sum | 62.842% | 71.387% | +8.545 points |
| Checksum | 59.619% | 68.750% | +9.131 points |

All seven required fields improve. Field outcomes from the same task are
dependent, so these rows are descriptive and are not treated as independent
samples.

### Confirmation suite and runtime contract

Version 2 of the program-execution generator creates modular-arithmetic loops
over four registers, a branch, an emitted sequence, a rotation, an
odd-position sum, and a checksum. The standard, demanding, and stress strata
use 8, 20, and 32 loop iterations.
The complete suite has SHA-256
`3b96e8869f63ac7893b2392c9b81b732d1646fe3737baa4ca37644aab6d897ea`;
block A has
`a0191b4f1afd6bccaa53bb5043c200da75f7fc7d3d26b8b43f99709edb587333`;
block B has
`a14a0759ef3b8642a62c01058d1a17965f5d8539121ecca735a3c7902aa6220b`.

Each block uses one TP4 instance per checkpoint with image
`voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`,
96 scheduled sequences, client concurrency 96, BF16 activations, a
28-GiB-per-GPU FP8 KV-cache reservation, a 4,096-token scheduler budget, a
65,536-token model limit, CUDA graph capture through 96 sequences, and no MTP
or speculative decoding. Sampling uses temperature zero, maximum reasoning
effort, a 32,768-token output limit, and request-seed base `20260907`. The
normalized runtime-contract hashes are
`c51e5870f67ab9f945155fe37dd9df7157c96cb4ccfa5c23599be8e334b1e378`
for block A and
`3d8367a16eee3b9a0de51a2733d2ebed89a7c4b52b907ca643b0c150bc2bd18c`
for block B; reference and candidate match within each block.

All 4,096 model responses completed without an API error. An independent
verifier re-executed every prompt, recomputed every seven-field score, and
validated every receipt digest. It found zero answer-key, score, digest,
missing-receipt, duplicate-receipt, or unexpected-receipt discrepancies.

### Research-only concurrency sensitivity

An incomplete capacity-preflight artifact retains repeated observations for
145 published-NVFP4 tasks in block A and 99 QAD tasks in block B under
32 scheduled sequences and client concurrency 32. Comparing those same
checkpoint/task pairs with their concurrency-96 responses gives:

| Checkpoint | Repeated tasks | Identical parsed answers | Score at concurrency 32 | Score at concurrency 96 | Difference |
|---|---:|---:|---:|---:|---:|
| Published NVFP4 | 145 | 74 (51.0%) | 75.665% | 70.443% | -5.222 points |
| QAD step 2,500 | 99 | 55 (55.6%) | 76.623% | 76.912% | +0.289 points |

Status: **research-only**. The subsets are incomplete, small, selected by
completion before interruption, and cover different task blocks and
checkpoints. They cannot estimate a checkpoint contrast or a causal
concurrency effect. They do demonstrate that temperature-zero generation is
not bit-identical across these scheduler-capacity configurations. The
qualified comparison remains valid because reference and candidate use the
same concurrency-96 contract within each counterbalanced block.

## What the result says about KLD training

The [QAD step-2,500 distribution-fidelity report](../../kld/glm-5.3-flash-qad-step2500.md)
shows that QAD step 2,500 reduces held-out natural-route Kullback-Leibler
Divergence (KLD) by 20.373% and exact-BF16-route KLD by 5.054% relative to
published NVFP4. Both predeclared KLD intervals exclude zero.

The aggregate VBF result establishes a different fact: that large
distribution-fidelity gain preserves the task-weighted semantic score within
±1 point on 9,856 deterministic tasks. It does not demonstrate an aggregate
behavioral improvement, and the exact-task and family diagnostics are not
uniformly favorable. The qualified 2,048-task program-execution comparison
does establish a narrow behavioral gain on the same generated execution
template, but it does not show that lower KLD caused that gain or that the gain
transfers to other capabilities. Lower KLD is therefore evidence that the
student better matches the teacher distribution under the KLD contract; it is
not by itself a monotonic measure of downstream answer quality.

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

The [aggregate machine-readable summary](validation/qad-step2500-vbf-pooled-9856-20260907.json)
and [program-execution machine-readable summary](validation/qad-step2500-program-execution-confirmation-2048-20260907.json)
contain full-precision values and component provenance.

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
| 2,048-task program-execution contract | SHA-256 `c4a41fe65f98423bdfdc9949f82f70f8a7bb74e1c3908f0d50d6e3a999f021f2` |
| Concurrency-96 runtime amendment | SHA-256 `2833fbc490119190cb56870b15494853a951353debea182a8c65ca95f6b7b797` |
| Program-execution preregistration manifest | SHA-256 `b352981f2c9eba72d4bd2efd224b19eea5caecd7b8754e35d14108c665586afd` |
| Program-execution runs | Block A published run `2026-09-07-765ff81671b56e9d`, receipt SHA-256 `829836b83f7931ce432617915c46ff0822b56f9c7730dd7b2bc4b3623345effa`; block A QAD run `2026-09-07-e4b99139e40076f7`, receipt SHA-256 `f4f926ff573898fea794478d3bd58a7b749e3d1ca3add65baad214e4bc4186e6`; block B published run `2026-09-07-1fbb0ffd6e988552`, receipt SHA-256 `c5086a25c6dd1eb739935534202ca39addacc5011594eee50c266565a762ccdd`; block B QAD run `2026-09-07-74fedb2cc3db0318`, receipt SHA-256 `54bd41a371bd5df184f414ff0e5efdb3b2e73785be53adfd905afd59f9bce365` |
| Program-execution analysis | JSON SHA-256 `4e4bdeb3115840ebfa22e5e5afa21841962cbe3a9bf2744267602c0eb366ecc7`; Markdown SHA-256 `7336b16c586570ea2748cd5e9e85f4323671c1cfe1bebd9e94f418fcd294a574` |
| Independent program-execution verification | Block A published SHA-256 `df8d7fe75803a1cec593f9df569dfd7f8879046139735171add1fc308ff70453`; block A QAD `4079ce221d31d60cb52af7a54fecc42614b9307519e1c1070f3304ff8dacd090`; block B published `780fb6f3bbd3ec5865f6e7d588136d4d0cce7828ba9470fa8035ac260dda7423`; block B QAD `90184a02a6a713684c224e7d2b51287667e2acf8930ec81bfebfd25c7d3e1536` |
| Program-execution public summary | SHA-256 `4e4bdeb3115840ebfa22e5e5afa21841962cbe3a9bf2744267602c0eb366ecc7` |
| Concurrency-sensitivity inputs | Published-NVFP4 concurrency-32 receipts SHA-256 `54353d76caca2a8bf8cf6fef7500e7b440cb0a49502c19d0b5ae5b435fddaf72`; QAD concurrency-32 receipts `8be57421e0ebc7c091971d7cef06c3f6ff37eab6e23de5960e584f4e82fd3f6f` |
| Program-execution retained artifact | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/nvfp4-qad-step2500-program-execution-confirmation-2048-20260907`; checksum-manifest SHA-256 `a6e132caf2cd451e388efe69492ba7158d047c9a799b848078db37ae27d601c4` |

## Attribution

The deterministic task generators, executable answer keys, strict scoring
contract, durable receipt format, runtime-verification records, component
executions, pooled analysis, and report are Local Inference Lab work. Paired
bootstrap intervals and McNemar's exact test are established statistical
methods rather than claimed mathematical inventions.
