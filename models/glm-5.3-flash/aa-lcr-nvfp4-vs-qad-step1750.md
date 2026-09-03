# GLM-5.3-Flash AA-LCR: published NVFP4 versus QAD step 1,750

Status: **qualified**.

This report compares two GLM-5.3-Flash checkpoint-and-serving configurations
on the pinned Artificial Analysis Long Context Reasoning (AA-LCR) dataset:

- published `local-inference-lab/GLM-5.3-Flash-NVFP4` revision
  `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`; and
- Local Inference Lab Quatrain quantization-aware-distillation checkpoint
  `GLM-5.3-Flash-NVFP4-QAD-step1750`.

Each configuration generated three independent answers for every one of the
100 dataset questions. GPT-5.6 Luna at medium reasoning classified the same
600 answers with the frozen AA-LCR equality-checker prompt.

The evaluated QAD configuration is 1.00 percentage point below the published
NVFP4 configuration, but the paired evidence does not distinguish their
AA-LCR accuracy. The exact two-sided McNemar p-value is `0.7608`, and the
question-cluster bootstrap interval includes both practically relevant
directions.

## Result

| Checkpoint and serving configuration | Correct | AA-LCR pass@1 reproduction |
|---|---:|---:|
| Published NVFP4, MTP depth 3, TP4/DCP1 | 222/300 | **74.00%** |
| QAD step 1,750, MTP depth 3, TP4/DCP1 | 219/300 | **73.00%** |
| QAD minus published NVFP4 | -3/300 | **-1.00 percentage point** |

Per-repeat counts show ordinary free-running variation rather than a monotonic
difference:

| Checkpoint | Repeat 0 | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| Published NVFP4 | 76/100 | 74/100 | 72/100 |
| QAD step 1,750 | 73/100 | 71/100 | 75/100 |

The paired 300-attempt contingency table contains 199 both-correct pairs, 23
published-NVFP4-only correct pairs, 20 QAD-only correct pairs, and 58
both-incorrect pairs. The exact two-sided McNemar p-value is
`0.7607916426`.

A 200,000-replicate bootstrap resampled the 100 questions while retaining all
three observed attempts belonging to each sampled question. Its 95% interval
for QAD minus published NVFP4 is **-5.33 to +3.33 percentage points**. The
bootstrap used seed `20260903`. It estimates variation across questions for
the observed generations and labels; it does not include additional
free-running generation or equality-checker repeat variation.

The machine-readable paired receipt is
[`aa-lcr-nvfp4-vs-qad-step1750-luna-20260903.json`](validation/aa-lcr-nvfp4-vs-qad-step1750-luna-20260903.json).
Its SHA-256 is
`63d335bb0189ec6f9c38c8fbd83535a8de5833b24a3cebded27100f13a7d93df`.

## Relationship to distribution fidelity

The [route-controlled distribution-fidelity report](../../kld/glm-5.3-flash-qad-step1750.md)
finds lower teacher-relative divergence for QAD step 1,750. On 524,020 held-out
next-token positions, QAD reduces forward Kullback-Leibler divergence by
10.001% under exact BF16-route replay and by 10.804% under natural routing.

The AA-LCR result does not show a corresponding capability gain. This is not a
contradiction:

- distribution fidelity measures teacher-forced next-token probabilities over
  the full vocabulary on fixed token sequences;
- AA-LCR measures stochastic, autoregressive, long-form answers to 100 tasks;
- the distribution-fidelity gain is heterogeneous and concentrated in one
  allocation stratum; and
- a one-percentage-point AA-LCR difference is much smaller than the uncertainty
  supported by 100 question clusters.

The supported conclusion is narrow: QAD step 1,750 is closer to the BF16
teacher under the declared distribution tests, while this AA-LCR sample does
not establish that either evaluated serving configuration has higher
long-context reasoning accuracy.

## Dataset and generation contract

| Property | Qualified value |
|---|---|
| Dataset | `ArtificialAnalysis/AA-LCR` |
| Dataset revision | `bdae010bbce259820c0e34c1d7cce210d966fb75` |
| Questions | 100 |
| Independent generations per question | 3 |
| Prompt messages | one user message; no system message |
| Reasoning effort | maximum |
| Temperature / top-p | `1.0` / `0.95` |
| Request seed | omitted |
| Maximum output | 163,840 tokens |
| Streaming | disabled |
| Repeat scheduling | all three repeats for one question run serially on one endpoint |
| Completion status | all 600 responses ended with `stop` |

The 163,840-token ceiling and sampling settings follow the GLM-5.3-Flash
benchmark recommendations. A ceiling is not a requested length: the published
NVFP4 responses used a median of 2,030 completion tokens and a maximum of
56,886; QAD responses used a median of 1,870 and a maximum of 50,085.

The pinned tokenizer produces 76,820 to 114,611 input tokens after applying
the GLM chat template, with a median of 100,972. The
[token-count manifest](validation/aa-lcr-glm-token-counts-378ca545.json) has
SHA-256
`6b5b4b3fff2b3cf0179591c3ee1721474dd588dea6504031caa22fb856509562`.
Both checkpoint runs report exactly the pinned per-question prompt counts.

## Serving configurations

Every serving replica used the same per-replica configuration:

| Property | Value |
|---|---|
| Hardware | four NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs |
| Tensor / decode-context parallelism | TP4 / DCP1 |
| Target activation / KV cache dtype | BF16 / FP8 |
| Target attention / KDA prefill | B12X / FlashKDA |
| Routed experts / linear layers | B12X / B12X |
| Weight loader | InstantTensor |
| Speculation | MTP depth 3; probabilistic draft sampling; standard rejection sampling |
| MTP attention / routed experts | B12X / Marlin |
| Maximum model length | 1,048,576 tokens |
| Scheduler token budget | 4,096 |
| Maximum active sequences | 24 |
| Explicit KV allocation | 30 GiB per GPU |
| Physical KV capacity | 4,621,648 tokens per replica |
| Prefix caching | enabled |
| CUDA graphs | full and piecewise through capture size 256 |

The immutable container digest is
`voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340`.
The embedded source-lock SHA-256 is
`9a6167d415d824e1707ba7df0dd5906e05c004f1ed2666f80f2f9e1ea9fde4be`.
It identifies vLLM commit
`7015eb6949a93247df02fb6f9101d17c40bd83e8` and B12X commit
`1e59a1fd09f782d302b1068b15c8a0bd66103894`.

The published NVFP4 pool used two identical TP4 replicas and 48 total client
workers. The QAD pool used three identical TP4 replicas and 72 total client
workers. Question IDs were assigned modulo the endpoint count. The additional
QAD replica shortened wall-clock execution but means the comparison qualifies
two complete checkpoint-and-runtime pools rather than isolating checkpoint
weights under an identical global scheduler history. Each response is an
unseeded independent sample, so no claim of bitwise paired generation is made.

The retained public runtime receipts are:

- [published NVFP4 runtime](validation/aa-lcr-nvfp4-runtime-20260903.json),
  SHA-256
  `cf99c4d44c3c7128b778c691694540ecb1c1289db29ac1b82f098def410a7134`;
- [QAD step 1,750 runtime](validation/aa-lcr-qad-step1750-runtime-20260903.json),
  SHA-256
  `aac82d4c21b74826d0a1672c1783b805abbe73d6fcb587de8082f6e3f7613f1c`.

## Equality checker

Artificial Analysis methodology version 4.1.1 specifies GPT-5.6 Luna at medium
reasoning for AA-LCR equality checking. The qualified runs reproduce that
model and reasoning level through authenticated Codex CLI version `0.152.1`.
The checker receives the question, official answer, and candidate answer, but
not the candidate checkpoint identity. It must emit only `CORRECT` or
`INCORRECT`.

The judge ran in a fresh empty workspace for every attempt, with user
configuration disabled and a read-only sandbox. Four judge requests ran
concurrently. Every expected receipt is present and no failure sidecar exists.

This locally executed reproduction is **not an official Artificial Analysis
leaderboard score**. It uses the disclosed methodology and pinned public data,
but it did not run inside Artificial Analysis infrastructure. The methodology
source is
[Artificial Analysis Intelligence Benchmarking Methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking).

## Qualification receipts

| Artifact | Published NVFP4 SHA-256 | QAD step 1,750 SHA-256 |
|---|---|---|
| Checkpoint weight index | `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb` | `b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b` |
| Runtime manifest | `cf99c4d44c3c7128b778c691694540ecb1c1289db29ac1b82f098def410a7134` | `aac82d4c21b74826d0a1672c1783b805abbe73d6fcb587de8082f6e3f7613f1c` |
| Generation manifest | `80217f2fc90a8bd19224b081edba4e05c8dd5c98ecca2c5fa159676b4260e89c` | `e3c45e98b92935834533b4831783409260e87de1775958c25099308cec4a6faa` |
| Generation completeness | `7307c6201083b87edc1690cca26dbc76570ccbbc55e1be7d0510f29de53366f7` | `a4227af03471f8b237edfa2dccb574d680259e654fb47fc1f300638e10823e81` |
| Judge manifest | `3a91552385c5df78dff3da8187011fe82d49e2bba7cc90a0c416a37765f42313` | `1e8b2ac1d553a0d8564859cfb621a92611799c332c177c0ecd76fe986196bec2` |
| Judge summary | `abc378076e3899c673457c5cfd1070eacb6575181a736df9091fbd6359b7fbc7` | `19cfa3dac2ede80cce6fb3a3ec2f42b0cbeafc25e7a6b02691f784cc9dfa220a` |
| Judge receipt-set canonical hash | `d91503e0266d51d6e1b6ce3286cd080c27a95866e5764e616b3723f7e1c6e98c` | `3f345c207bb2c4777db720607e9a5d5cc51ef9bc14614d27e2be55976767c9e6` |

The QAD complete checkpoint file-manifest SHA-256 is
`12362645b613f625f5e7bc008050db51181cb3b50fcc5e066b97a8494f9fcf33`.
Its materialization-manifest SHA-256 is
`4c8439a5e8891f0dc7f22134803e44aebd4296284e356ae0a8bb2cf76e4ee937`.

The public validation directory contains aggregate and configuration
receipts, not the 600 candidate answers or 600 per-attempt judge receipts.
Local Inference Lab retains those full receipt sets. The aggregate claims are
qualified against the retained data; third parties can audit the published
contracts and hashes but cannot independently recompute every label from the
wiki alone.

See the [AA-LCR reproduction specification](aa-lcr-reproduction.md) for the
prompt, launch, generation, validation, judging, and comparison procedures.

## Interpretation limits

AA-LCR scores depend on the checkpoint, runtime, sampling configuration,
equality checker, and finite question sample. They are not checkpoint-only
constants. In particular:

- three unseeded generations do not measure all free-running variation;
- the equality checker was not independently repeated;
- the paired bootstrap covers question sampling, not generation or judge
  resampling;
- pool sizes and global scheduler histories differ;
- the QAD checkpoint is a local research artifact rather than a published,
  qualified serving target; and
- no throughput conclusion may be drawn from the generation wall times because
  the pool sizes differ.

The QAD configuration remains **research-only** as a deployment target. The
AA-LCR artifact is **qualified** for the exact checkpoint, runtime, dataset,
sampling, and equality-checker identities reported above.
