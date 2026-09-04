# GLM-5.3-Flash AA-LCR: BF16, published NVFP4, and QAD step 1,750

Status: **qualified**.

This report compares three GLM-5.3-Flash checkpoint-and-serving
configurations on the pinned Artificial Analysis Long Context Reasoning
(AA-LCR) dataset:

- `zai-org/GLM-5.3-Flash-BF16` revision
  `61f77a1e1a67c410650ce5017411337da0dcd11a`, served with TP8/DCP1;
- published `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`, served with TP4/DCP1; and
- Local Inference Lab Quatrain quantization-aware-distillation checkpoint
  `GLM-5.3-Flash-NVFP4-QAD-step1750`, served with TP4/DCP1.

Each configuration generated three independent answers for every one of the
100 dataset questions. Candidate generation used `reasoning_effort=max`.
GPT-5.6 Luna at `reasoning_effort=medium` classified all 900 retained answers
with the frozen AA-LCR equality-checker prompt.

The observed point ordering is published NVFP4, QAD step 1,750, then BF16.
The paired evidence does not distinguish BF16 from either NVFP4
configuration: both exact McNemar p-values exceed 0.40 and both
question-cluster bootstrap intervals include zero.

## Result

| Checkpoint and serving configuration | Correct | AA-LCR pass@1 reproduction |
|---|---:|---:|
| Published NVFP4, MTP depth 3, TP4/DCP1 | 222/300 | **74.00%** |
| QAD step 1,750, MTP depth 3, TP4/DCP1 | 219/300 | **73.00%** |
| BF16, MTP depth 3, TP8/DCP1 | 215/300 | **71.67%** |

The per-repeat counts are:

| Checkpoint | Repeat 0 | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| Published NVFP4 | 76/100 | 74/100 | 72/100 |
| QAD step 1,750 | 73/100 | 71/100 | 75/100 |
| BF16 | 73/100 | 71/100 | 71/100 |

## Paired comparisons

| Candidate minus reference | Difference | Reference only / candidate only | Exact two-sided McNemar p | Question-cluster bootstrap 95% interval |
|---|---:|---:|---:|---:|
| BF16 minus published NVFP4 | **-2.33 percentage points** | 29 / 22 | `0.4011` | **-7.33 to +2.67 points** |
| BF16 minus QAD step 1,750 | **-1.33 percentage points** | 21 / 17 | `0.6271` | **-5.00 to +2.33 points** |

The BF16-versus-published-NVFP4 contingency table contains 193 both-correct
pairs, 29 published-NVFP4-only correct pairs, 22 BF16-only correct pairs, and
56 both-incorrect pairs. The machine-readable receipt is
[`aa-lcr-published-nvfp4-vs-bf16-luna-20260904.json`](validation/aa-lcr-published-nvfp4-vs-bf16-luna-20260904.json),
with SHA-256
`33ca17043f05d0cb03216dcdbb0355d4a9507e4c6effcc4ee54f565e04f7330d`.

The BF16-versus-QAD contingency table contains 198 both-correct pairs, 21
QAD-only correct pairs, 17 BF16-only correct pairs, and 64 both-incorrect
pairs. The machine-readable receipt is
[`aa-lcr-qad-step1750-vs-bf16-luna-20260904.json`](validation/aa-lcr-qad-step1750-vs-bf16-luna-20260904.json),
with SHA-256
`6ca10ca0b5e59791f4c31d3ffd65145c668cd227be6850d2d62442d4fc97930e`.

Both bootstrap intervals use 200,000 replicates and seed `20260903`. Each
replicate resamples the 100 questions while retaining all three observed
attempts belonging to a sampled question. The intervals estimate variation
across questions for the retained generations and labels; they do not include
additional free-running generation or equality-checker repeat variation.

The separate
[published-NVFP4-versus-QAD report](aa-lcr-nvfp4-vs-qad-step1750.md)
records a QAD-minus-published-NVFP4 difference of -1.00 percentage point,
exact McNemar `p=0.7608`, and a bootstrap interval of -5.33 to +3.33 points.

## Dataset and generation contract

| Property | Qualified value |
|---|---|
| Dataset | `ArtificialAnalysis/AA-LCR` |
| Dataset revision | `bdae010bbce259820c0e34c1d7cce210d966fb75` |
| Questions | 100 |
| Independent generations per question | 3 |
| Prompt messages | one user message; no system message |
| Candidate reasoning effort | `max` |
| Temperature / top-p | `1.0` / `0.95` |
| Request seed | omitted |
| Maximum output | 163,840 tokens |
| Streaming | disabled |
| Repeat scheduling | all three repeats for one question run serially on one endpoint |
| BF16 completion status | 300 responses ended with `stop`; zero failure sidecars |

The pinned GLM tokenizer produces 76,820 to 114,611 input tokens after applying
the chat template, with a median of 100,972. The
[token-count manifest](validation/aa-lcr-glm-token-counts-378ca545.json) has
SHA-256
`6b5b4b3fff2b3cf0179591c3ee1721474dd588dea6504031caa22fb856509562`.
The BF16 tokenizer, tokenizer configuration, and chat template are bit-identical
to the published NVFP4 revision, and all 300 BF16 responses report the pinned
per-question prompt counts.

The BF16 responses contain 1,387,393 completion tokens. Their median is
1,813.5 tokens, their 95th percentile is 18,133.5, and their maximum is
47,741. The maximum output is below the declared ceiling, so no qualified
response was truncated by `max_tokens`.

## BF16 serving configuration

| Property | Qualified value |
|---|---|
| Hardware | eight NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs |
| Checkpoint weights | unquantized BF16, 120 safetensors shards |
| Tensor / decode-context parallelism | TP8 / DCP1 |
| Target activation / KV cache dtype | BF16 / FP8 DeepSeek-MLA |
| Target attention / KDA prefill | B12X / FlashKDA |
| Routed experts | FlashInfer CUTLASS unquantized |
| Weight loader | vLLM automatic safetensors loader |
| Speculation | MTP depth 3; probabilistic draft sampling; standard rejection sampling |
| Maximum model length | 300,000 tokens |
| Scheduler token budget | 2,048 |
| Maximum active sequences | 4 |
| Explicit KV allocation | 9 GiB per GPU |
| Physical KV capacity | 1,272,727 tokens |
| Prefix caching | enabled |
| CUDA graphs | full and piecewise at capture sizes 1, 2, and 4 |

The 300,000-token model limit exceeds the largest possible request under the
declared contract: 114,611 prompt tokens plus the 163,840-token output ceiling
is 278,451 tokens. The server reports 4.24-way KV capacity at the 300,000-token
limit.

The immutable container digest is
`voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`.
The embedded source-lock SHA-256 is
`9a6167d415d824e1707ba7df0dd5906e05c004f1ed2666f80f2f9e1ea9fde4be`.
It identifies vLLM commit
`7015eb6949a93247df02fb6f9101d17c40bd83e8` and B12X commit
`1e59a1fd09f782d302b1068b15c8a0bd66103894`.

The BF16 configuration used one TP8 replica and four client workers. The
published NVFP4 configuration used two TP4 replicas and 48 client workers; QAD
used three TP4 replicas and 72 client workers. The comparison therefore
qualifies complete checkpoint-and-runtime configurations. It does not isolate
checkpoint weight format under an identical topology, global scheduler
history, or serving capacity. No throughput conclusion may be drawn from the
generation wall times.

### Long-context memory qualification

The qualified BF16 server uses a 2,048-token scheduler budget and an explicit
9 GiB KV allocation. It completed all 300 responses, including all three
attempts for the 114,611-token maximum prompt, without an inference error.

A TP8 BF16 configuration with a 4,096-token scheduler budget and 0.95 automatic
GPU-memory allocation is **unsupported** for this AA-LCR workload. A
95,407-token preflight prompt reached the B12X multi-head-composition path but
failed before producing a candidate answer when temporary 32 to 128 MiB CUDA
allocations could not be satisfied. The container itself was not OOM-killed.
The machine-readable qualification receipt is
[`aa-lcr-bf16-bt4096-auto95-unsupported-20260903.json`](validation/aa-lcr-bf16-bt4096-auto95-unsupported-20260903.json),
with SHA-256
`08b44380b7b25f1b26c7853589e667551f4b5e3b623470d90a99c30f92f4269f`.

## Equality checker

Artificial Analysis methodology version 4.1.1 specifies GPT-5.6 Luna at medium
reasoning for AA-LCR equality checking. The qualified BF16 run reproduces that
model and reasoning level through authenticated Codex CLI version `0.152.1`.
The checker receives the question, official answer, and candidate answer, but
not the candidate checkpoint identity. It must emit only `CORRECT` or
`INCORRECT`.

Every answer ran in a fresh empty workspace with user configuration disabled
and a read-only sandbox. Four judge requests ran concurrently. All 300 expected
receipts are present and no failure sidecar exists.

This locally executed reproduction is **not an official Artificial Analysis
leaderboard score**. It uses the disclosed methodology and pinned public data,
but it did not run inside Artificial Analysis infrastructure. The methodology
source is
[Artificial Analysis Intelligence Benchmarking Methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking).

## BF16 qualification receipts

| Artifact | SHA-256 |
|---|---|
| BF16 checkpoint weight index | `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f` |
| [Runtime manifest](validation/aa-lcr-bf16-runtime-20260903.json) | `fa457e262698fb12c0f9ac3680beced879f488ab9059b572dfbf1f5684589a57` |
| [Generation manifest](validation/aa-lcr-bf16-generation-manifest-20260903.json) | `456822fe9b53e893ed789cdfe4f88597cb91938bdc6d29edf9216291b42288a7` |
| [Generation completeness](validation/aa-lcr-bf16-generation-completeness-20260904.json) | `de5357f136c0f888068806f08a99d85b57271e302f8cd32a59ea7ed1cb20b661` |
| [Judge manifest](validation/aa-lcr-bf16-luna-judge-manifest-20260904.json) | `bccba34d15ac51bf8471faeadc7f2205af9b5ba5faef0b5a65a71f0760a74c4b` |
| [Judge summary](validation/aa-lcr-bf16-luna-summary-20260904.json) | `63ec7b2f6c63e18c3eb202d872177d80cf6611cbdbf11dcfa9f1cce6ae0af8ef` |
| Judge receipt-set canonical hash | `6809e3b050f3568b91c65653de0f13903768ce878310d3381845534cbb88175a` |

The public validation directory contains aggregate and configuration receipts,
not the 300 BF16 candidate answers or 300 per-attempt judge receipts. Local
Inference Lab retains those full receipt sets. The aggregate claims are
qualified against the retained data; third parties can audit the published
contracts and hashes but cannot independently recompute every label from the
wiki alone.

See the [AA-LCR reproduction specification](aa-lcr-reproduction.md) for the
prompt, generation, validation, equality-checking, and comparison procedures.

## Interpretation limits

AA-LCR scores depend on the checkpoint, runtime, sampling configuration,
equality checker, and finite question sample. They are not checkpoint-only
constants. In particular:

- three unseeded generations do not measure all free-running variation;
- the equality checker was not independently repeated;
- each paired bootstrap covers question sampling, not generation or judge
  resampling;
- the three serving pools have different topology, capacity, and scheduler
  history; and
- the QAD checkpoint is a local research artifact rather than a published,
  qualified serving target.

The supported conclusion is narrow: the point estimates are 74.00% for
published NVFP4, 73.00% for QAD step 1,750, and 71.67% for BF16, while this
AA-LCR sample does not establish that any one of the three evaluated complete
configurations has higher long-context reasoning accuracy.
