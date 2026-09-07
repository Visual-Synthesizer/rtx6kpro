# GLM-5.3-Flash verifier-backed behavioral fidelity

Status: **qualified execution; inconclusive one-percentage-point decision**.

Verifier-Backed Behavioral Fidelity (VBF) is a Local Inference Lab regression
benchmark for a practical question: when checkpoint weights change, do the
changes produce objectively better or worse answers?

VBF is a project-specific benchmark with a published operation contract, not
an established external leaderboard or a universal model-quality score.

The [QAD step-2,500 companion report](qad-step2500-verifier-backed-behavioral-fidelity.md)
applies VBF to 9,856 non-overlapping task pairs for published NVFP4 and QAD
step 2,500. Its primary result establishes practical equivalence inside a
predeclared ±1-point semantic-score margin. A separate 224-task TP4×2
execution remains variance-estimation evidence and retains the QAD step-1,750
comparison.

VBF gives each checkpoint the same deterministic tasks and scores every answer
against programmatically computed ground truth. It does not ask another model
to judge the answer, and it does not compare candidate prose with a BF16
generation. A quantized checkpoint can therefore receive credit when it solves
a task that BF16 misses, and it receives a regression when BF16 is correct and
the quantized checkpoint is not.

The evaluated 224-task execution is complete and satisfies the recorded
checkpoint-comparison contract. Its statistical resolution is not sufficient
to establish improvement, non-inferiority, or equivalence inside the declared
one-percentage-point margin.

## Result in plain language

| Checkpoint | Semantic score | Completely correct tasks | Output protocol valid |
|---|---:|---:|---:|
| BF16 reference | **93.11%** | **198/224 (88.39%)** | 97.32% |
| Published NVFP4 | 91.63% | 192/224 (85.71%) | 95.98% |
| QAD step 1,750 | 92.35% | 190/224 (84.82%) | 97.32% |

Quantization-Aware Distillation (QAD) step 1,750 is closer to BF16 than the
published NVIDIA 4-bit floating-point (NVFP4) checkpoint on the primary
fractional score: its observed loss is 0.77 percentage points instead of 1.49
points. That is encouraging, but it is not a demonstrated quality improvement.

The paired uncertainty intervals are several points wide. QAD also has the
lowest exact-task point estimate, despite its better fractional score. The
supported conclusion is therefore:

- QAD changes useful behavior, with both regressions and recoveries;
- the observed aggregate VBF score is closer to BF16 than published NVFP4 is;
- the retained data do not establish that QAD is better, worse, equivalent to,
  or non-inferior to BF16 within one percentage point; and
- QAD remains **research-only as a claimed behavioral improvement**.

`Inconclusive` is not a euphemism for failure and is not proof of equivalence.
It means the completed sample does not contain enough information for the
declared decision threshold.

## What VBF measures

Kullback-Leibler Divergence (KLD) measures whether the complete next-token
probability distribution moved. VBF measures whether model behavior moved
across the boundary between objectively correct and incorrect answers.

For example, a small distribution change can leave the final answer unchanged,
while another small change can alter one digit in a total, reverse a rule
precedence decision, or select the wrong node in a dependency graph. KLD sees
both distribution changes; VBF distinguishes their observed task outcomes.

VBF is intentionally narrow. It measures deterministic reasoning and
instruction execution when all required facts are supplied in the prompt. It
does not measure creative writing, style, subjective usefulness, factual
knowledge not present in the prompt, safety, tool use, or every possible user
workload. It complements KLD and capability-specific evaluations rather than
replacing them.

## Test construction

Python generators create both the prompt and its answer key. No language model
writes the prompts or computes the expected answers. A master seed, task family,
and item number deterministically select every generated value.

The suite contains 32 prompts from each of seven equally represented task
families. Each family cycles through standard, demanding, and stress inputs.

| Task family | Behavior being tested |
|---|---|
| Record reconciliation | Apply ordered corrections, filter records, join payments, and calculate exact aggregates. |
| Event-sourced state | Reconstruct mutable state from an ordered event stream and answer final-state queries. |
| Dependency graph | Calculate reachability, path counts, shortest and longest paths, ancestry, and mandatory nodes in a directed acyclic graph. |
| Constraint assignment | Solve a one-to-one ordering problem that the generator has verified has one unique solution. |
| Program execution | Execute precisely specified integer control flow without executing model-produced code. |
| Policy application | Apply business rules with explicit priority, precedence, and boundary operators. |
| Evidence-chain retrieval | Follow corrected asset, material, and supplier relationships through distractor-heavy context up to approximately 15,000 characters. |

Every prompt requests one JSON object with named fields and explicit value
types. The generated JSONL suite contains a manifest, immutable task records,
and a SHA-256 digest over the canonical task content. Loading stops on a changed
record, duplicate task identifier, unsupported schema, or invalid answer
contract.

The answer key is retained beside each task for scoring but is never included
in the request sent to the model.

## How one answer is scored

The scorer removes separately returned reasoning and inline `<think>` blocks,
then finds the last complete JSON object in visible answer content. It does not
repair malformed JSON, convert strings to numbers, treat booleans as integers,
ignore list order, or guess an intended answer.

Suppose the expected object is:

```json
{
  "eligible_count": 3,
  "overdue_ids": ["INV-002", "INV-009"],
  "outstanding_total": 950
}
```

This answer has two of three fields correct because the list order is wrong:

```json
{
  "eligible_count": 3,
  "overdue_ids": ["INV-009", "INV-002"],
  "outstanding_total": 950
}
```

VBF retains two scores for every task:

1. **Semantic score** is the fraction of required top-level fields whose value
   and type exactly match the answer key. Each prompt receives equal aggregate
   weight, regardless of its number of fields.
2. **Exact-task accuracy** is one only when every required field is correct and
   the object contains exactly the required keys. A missing or extra key makes
   the exact score zero.

Semantic score is the primary metric because it preserves information about
partial damage. Exact-task accuracy provides the stricter user-visible view.
Per-field counts are retained as diagnostics, but fields belonging to one
prompt are not treated as independent statistical samples.

## Paired comparison

The comparison operates task by task. For each task, it subtracts the BF16
semantic score from the candidate semantic score, then averages those paired
differences. This answers whether the *same questions* changed; comparing two
unpaired accuracy intervals would not answer that question.

The report includes:

- a 95% paired bootstrap interval that resamples whole task identifiers;
- exact tasks solved only by the reference or only by the candidate;
- a two-sided exact McNemar test over exact-task flips;
- correct-to-incorrect field regressions and incorrect-to-correct recoveries;
- tasks where both models are wrong in different ways; and
- separate diagnostic intervals for every task family.

If a task has repeated generations, its repeats remain in one statistical
cluster. The evaluated GLM-5.3-Flash suite uses one generation per task.

## Decision rule

The practical margin was fixed at one semantic-score percentage point before
the comparison was interpreted. For candidate minus reference:

| Decision | Required paired 95% interval |
|---|---|
| `worse` | Entire interval is below -1 point. |
| `better` | Entire interval is above +1 point. |
| `practically_equivalent` | Entire interval lies between -1 and +1 points. |
| `not_worse` | Harm beyond -1 point is excluded, but two-sided equivalence is not established. |
| `inconclusive` | Interval still includes harm beyond -1 point. |
| `unsupported` | Complete pairing, provenance, or matched execution conditions are absent. |

One percentage point is a deployment-policy tolerance, not a mathematical
constant. A deployment that tolerates a different loss must declare its margin
before evaluating an independent suite.

## Checkpoints and matched execution

| Role | Artifact identity |
|---|---|
| BF16 reference | `zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a` |
| Published comparator | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |
| QAD candidate | `GLM-5.3-Flash-NVFP4-QAD-step1750`; Quatrain training step 1,750 |

All three checkpoints used:

- one user message per prompt and no system message;
- `temperature=0`, a task-derived request seed, and maximum reasoning effort;
- a 32,768-token output limit and one generation per task;
- eight NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs with Tensor
  Parallelism 8 and Decode Context Parallelism 1;
- Multi-Token Prediction disabled, eight concurrent sequences, a 65,536-token
  model limit, and a 4,096-token scheduler budget;
- FP8 key/value cache with 9 GiB allocated per GPU; and
- container image
  `voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`.

The normalized runtime-comparison contract SHA-256 is
`f007d372af980c07d4cfcd1fa2dc870415123e06e54aa9744f7c8a28f6d27856`
for every checkpoint. Weight-format-specific loaders and kernels differ where
the representation requires them; the remaining recorded serving inputs match.

A run is **qualified** only when the full suite executes, at least 32 tasks are
present in every family, no request fails, task pairing is complete, the suite
and receipts pass hash validation, and both runtime manifests contain the same
checkpoint-comparison contract. Qualification certifies provenance and
completeness; it does not guarantee enough statistical power for a directional
decision.

## Paired results

| Candidate minus reference | Semantic difference | Paired 95% interval | One-point decision | Exact harm / recovery | McNemar p |
|---|---:|---:|---|---:|---:|
| Published NVFP4 minus BF16 | **-1.49 points** | **-5.63 to +2.64** | `inconclusive` | 25 / 19 | `0.4514` |
| QAD minus BF16 | **-0.77 points** | **-4.42 to +2.85** | `inconclusive` | 28 / 20 | `0.3123` |
| QAD minus published NVFP4 | **+0.72 points** | **-3.18 to +4.66** | `inconclusive` | 21 / 19 | `0.8746` |

“Exact harm” means that the row's reference solved the complete task and the
candidate did not. “Recovery” means the reverse. QAD's semantic score is 0.72
points above published NVFP4, while its exact-task rate is 0.89 points lower.
Neither difference is resolved by the retained sample.

Relative to BF16, published NVFP4 contains 121 correct-to-incorrect field
changes and 98 recoveries across 51 tasks with at least one changed value. QAD
contains 92 field regressions and 86 recoveries across 54 tasks with a changed
value. These field totals describe where answers moved; they are not additional
independent samples.

## Task-family diagnostics

The family results show that aggregate similarity can hide opposing changes.
These intervals are exploratory: seven families were inspected and no
multiple-comparison correction was applied.

| Task family | QAD minus BF16 | Paired 95% interval | QAD minus published NVFP4 | Paired 95% interval |
|---|---:|---:|---:|---:|
| Constraint assignment | +0.00 points | +0.00 to +0.00 | +0.00 points | +0.00 to +0.00 |
| Dependency graph | +2.68 points | -7.59 to +13.84 | +6.70 points | -5.36 to +19.20 |
| Event-sourced state | -10.94 points | -25.00 to +2.34 | -10.16 points | -21.88 to -0.78 |
| Evidence-chain retrieval | +7.29 points | +0.26 to +16.93 | +5.47 points | -1.04 to +14.84 |
| Policy application | +0.00 points | +0.00 to +0.00 | +0.00 points | +0.00 to +0.00 |
| Program execution | +2.23 points | -10.71 to +15.18 | +2.23 points | -14.73 to +19.20 |
| Record reconciliation | -6.64 points | -15.62 to +1.95 | +0.78 points | -9.77 to +11.72 |

QAD has a higher evidence-chain-retrieval point estimate and a lower
event-sourced-state point estimate in this suite. Those observations are useful
hypotheses, not qualified domain claims. They require a separately frozen
replication suite.

Constraint assignment and policy application are at a 100% BF16 ceiling.
BF16 reaches a 35.71% semantic-score floor in the program-execution stress
stratum. No task or stratum was removed after candidate results were observed.

## Greedy-decoding noise control

A separate 42-task BF16 self-comparison is **research-only** because it contains
only six tasks per family. Two executions of the same BF16 checkpoint score
95.24% and 90.69%; the paired difference is -4.55 points with a 95% interval of
-10.80 to 0.00 points and three exact-task flips.

Greedy decoding, fixed request seeds, and a matched runtime therefore do not
eliminate batching and GPU-kernel nondeterminism. The 224-task execution uses
one run per checkpoint, so small observed differences cannot be attributed
solely to quantization.

The observed standard deviation of QAD-minus-BF16 task differences is 0.277. A
normal approximation requires approximately 2,950 independent tasks merely to
reduce the 95% interval half-width to one percentage point. Demonstrating
equivalence can require more because the observed difference is not centered
at zero. That estimate is planning guidance, not a guaranteed sample size.

## Relationship to KLD and AA-LCR

The separate [route-controlled distribution-fidelity report](../../kld/glm-5.3-flash-qad-step1750.md)
finds that QAD reduces held-out forward KLD relative to published NVFP4 by
10.00% under exact BF16-route replay and by 10.80% under natural routing. Both
predeclared directional criteria pass.

That distributional improvement does not imply the same ordering in task
accuracy. VBF finds an encouraging fractional point estimate for QAD but no
statistically resolved behavioral improvement. A separate 200-question
MMLU-Pro cross-check scores BF16 at 87.0%, published NVFP4 at 84.5%, and QAD at
84.0%; its paired tests also do not distinguish the checkpoints.

The [AA-LCR comparison](aa-lcr-bf16-vs-nvfp4.md) scores published NVFP4 at
74.00%, QAD at 73.00%, and BF16 at 71.67%. Its paired intervals include zero,
and its MTP depth 3 serving configurations differ from the MTP-disabled,
topology-matched checkpoint isolation used by VBF.

The combined evidence supports one narrow conclusion: QAD is measurably closer
to the BF16 next-token distribution than published NVFP4 on the held-out KLD
corpus, but no completed capability evaluation demonstrates a user-visible
quality improvement.

## Reproducibility record

| Evidence | Durable identifier |
|---|---|
| Suite | 224 tasks; 32 per family |
| Canonical task-record SHA-256 | `e6d71089c599db57f77b894ab693186100d81d8d15ad35017e8fc6870eac859d` |
| Complete suite-file SHA-256 | `67a7674a731dc9cd51697abe9c2bce9f3d406c1b211c6f947de1861004ac8b47` |
| Comparison receipt SHA-256 | `5b8a1b9b74002588566964a637d31719b89e1a9f6b644531a09ae79f73e0a6e6` |
| Bootstrap samples | 100,000 paired task-cluster resamples |
| Runtime-comparison contract SHA-256 | `f007d372af980c07d4cfcd1fa2dc870415123e06e54aa9744f7c8a28f6d27856` |
| Evaluated implementation | `local-inference-lab/llm-inference-bench`, package `behavioral_fidelity`, Git commit `db4e5d9f28ad66f769cef2ef2365e9d24dd7969b` |
| Retained artifact root | `/mnt/luke/evals/glm-5.3-flash-behavioral-fidelity/checkpoint-qualification-v2-224-20260904` |

The implementation writes an immutable run manifest, append-only hashed JSONL
receipts, and a derived summary. Resume mode skips successful task/repeat keys,
retries failures, and refuses a changed execution contract.

## Attribution

The deterministic task generators, strict JSON scoring contract, durable
receipt format, runtime matching, and VBF comparison implementation are Local
Inference Lab work. Paired bootstrap confidence intervals, Wilson intervals,
and McNemar's test are established statistical methods rather than claimed
mathematical inventions.
