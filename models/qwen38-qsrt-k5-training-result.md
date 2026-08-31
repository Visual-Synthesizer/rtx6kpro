# Qwen3.8-27B QSRT K5 recovery: what the training result means

This page explains the Qwen3.8-27B dense-MLP quantization experiment in plain
language. It separates the quality gained from activation-weighted singular
value decomposition (SVD) from the effect of subsequent gradient-based
Quantization-Aware Training (QAT), and it explains why the TensorBoard graphs
looked better while the broader evaluation selected the pre-training
checkpoint.

**Evidence snapshot:** 2026-08-21

**Status:** The packed QSRT K5 checkpoint with rank-16 Brain Floating Point
16-bit (BF16) recovery is a `qualified` research artifact. A general quality
improvement from gradient-based adapter QAT is `unsupported`. Production
replacement of the Microscaling 8-bit floating-point (MXFP8) serving tier is
`unsupported` because serving performance is not competitive.

The exact artifact, runtime, source, and test contracts are recorded in
[Qwen3.8-27B QSRT K5 dense-MLP recovery](qwen38-qsrt-k5-r16.md). This page is
the interpretation guide for those results.

## The result in one minute

- All 192 dense MLP matrices in the 64-layer text decoder were encoded as
  five-bit Quantile-Stratified Rate-shifted Trellis (QSRT K5) weights.
- Every quantized matrix received a rank-16 BF16 additive correction. The
  correction was initialized from activation-weighted SVD, not from random
  values.
- A 50-million-token, one-epoch QAT run propagated gradients through all 64
  layers. Only the 384 rank-16 correction tensors were trainable; the K5 base
  weights and every retained BF16 tensor stayed frozen.
- A small 56-document monitoring set showed a real reduction in mean
  Kullback-Leibler divergence (KLD) from `0.0147936` to `0.0107129`. This is
  the improvement visible in TensorBoard.
- A balanced 512-context evaluation of every saved optimizer boundary found no
  improvement over the SVD initialization. The best boundary was optimizer
  step 0, before the first AdamW update.
- A wider 5,120-context comparison confirmed that the trained step-1300
  correction was worse than the step-0 SVD correction.
- The selected artifact therefore validates QSRT K5 plus weighted-SVD
  recovery. It does not validate a general benefit from the subsequent
  Low-Rank Adaptation (LoRA) QAT updates.

The training process worked mechanically: gradients were finite, optimizer
updates changed the rank-16 factors, the complete corpus was processed, and
the monitored objective decreased. The unsupported conclusion concerns
generalization, not execution correctness.

## What was quantized and what was trained

The source model is
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The experiment
changes only the dense MLP path in the text decoder.

| Model component | Representation during the run | Trainable |
| --- | --- | --- |
| 64-layer decoder `gate_proj`, `up_proj`, and `down_proj` matrices | QSRT K5 | No |
| Rank-16 additive correction for each quantized matrix | BF16 factors with FP32 optimizer state | Yes |
| Attention and gated-DeltaNet projections | BF16 | No |
| RMSNorm, embeddings, vision tower, multi-token prediction (MTP) head, and `lm_head` | BF16 | No |

The corrected dense operation is:

```text
y = decode(W_QSRT_K5) x + B (A.T x)
```

`A` and `B` are the trainable rank-16 factors. Across 192 matrices, the run
contains 384 factor tensors and 69,206,016 trainable parameters. The model is
not an EXL3 or MCG checkpoint, and it is not a transcode of an MXFP8
checkpoint.

This is full-depth, adapter-only QAT. The forward and backward paths traverse
the complete decoder, but QAT does not update the decoded dense base weights,
RMSNorm gains, attention weights, or any other retained BF16 parameters. A run
that updates the dense base weights is a different and substantially more
expensive form of QAT.

## Training contract

The recovery corpus contains 35,790 complete documents from 11 sources and
exactly 50,000,000 stored tokens. The run processed 49,964,210 causal
positions in 1,680 optimizer groups.

| Setting | Value |
| --- | --- |
| Epochs | 1 |
| Optimizer | AdamW |
| Learning rate | `2e-5` with 100 warmup groups |
| Betas | `(0.9, 0.95)` |
| Compute | BF16 with FP32 masters, moments, and accumulated gradients |
| Hardware | Two RTX PRO 6000 GPUs for groups 0-399; four for groups 400-1679 |
| Wall time | 29.15 hours |

The training loss is full-vocabulary forward Kullback-Leibler divergence
(KLD) from the BF16 teacher distribution to the QSRT K5 plus rank-16 student
distribution. Both distributions use the same frozen BF16 language-model
head. Lower KLD means that the student's next-token distribution is closer to
the BF16 teacher on the measured token population.

## The three KLD signals

The run exposed several KLD graphs and reports. They answer different
questions and cannot be substituted for one another.

| Signal | Population | Purpose |
| --- | --- | --- |
| `train_batch/forward_mean_kl` | The documents in one optimizer group | Verify that gradient descent is optimizing the batches it sees |
| `held_out/mean_kl` and `held_out/rms_kl` in TensorBoard | 56 fixed documents, 65,480 causal positions | Cheap periodic monitoring during the run |
| All-boundary checkpoint selection | 512 balanced contexts, 1,048,064 causal positions | Select a correction that generalizes across sources |
| Wider population comparison | 5,120 contexts, 10,480,640 causal positions | Confirm the important checkpoint comparison at larger scale |

KLD is population-dependent. A checkpoint does not have one universal KLD
number. Document source, subject, length, token difficulty, and weighting all
change the average. Values such as `0.014` from the 56-document screen and
`0.003` from the balanced population must not be compared directly. Only
checkpoint differences measured on the same tokens are meaningful.

## Why TensorBoard showed an improvement

The TensorBoard `held_out/mean_kl` graph used a fixed 56-document screen. The
documents were excluded from gradient updates, so the `held_out` label is
technically accurate. The population was nevertheless source-concentrated and
too small to represent the broader target distribution.

Root mean square (RMS) KLD emphasizes large per-token divergences more strongly
than ordinary mean KLD.

| Optimizer boundary | Screen mean KLD | Screen RMS KLD | Interpretation |
| ---: | ---: | ---: | --- |
| 0 | 0.014793632 | 0.278722237 | Weighted-SVD initialization before AdamW |
| 1300 | **0.010712946** | 0.166408244 | Lowest mean KLD on the small screen |
| 1680 | 0.010830213 | **0.155067258** | End of the one-epoch run |

The step-1300 mean improved by 27.58% and its RMS improved by 40.30% relative
to step 0. The decrease displayed by TensorBoard is therefore a real
measurement, not a logging or smoothing error.

The screen also contained an early warning: step 1300 had p99 KLD
`0.132518`, which was 10.52% worse than the step-0 p99. The optimizer reduced
the average and some large outliers on this particular population without
producing a uniformly better distribution.

Because the same small screen was inspected repeatedly and used to identify
step 1300, it functions as a monitoring or validation set, not as an
independent final test. Its result establishes distribution-specific learning;
it does not establish broad quality improvement.

## What the balanced checkpoint selection found

Every durable boundary was evaluated again: optimizer step 0, every 25 groups,
and the final step, for 69 candidates in total. The selection population
contains 512 contexts, five equal-weight source strata, 366 source clusters,
and 1,048,064 causal positions. No context has an exact 12-token match with
the recovery corpus.

| Optimizer boundary | Mean KLD | p99 KLD | Top-1 agreement |
| ---: | ---: | ---: | ---: |
| 0 | **0.003167795** | **0.0319588** | 97.4637% |
| 25 | 0.003169655 | 0.0320798 | **97.4726%** |
| 50 | 0.003170315 | 0.0320911 | 97.4703% |
| 100 | 0.003192800 | 0.0321952 | 97.4565% |
| 1300 | 0.003308936 | 0.0331666 | 97.4247% |

Step 25 is the closest trained candidate. Step 0 has mean KLD lower by
`0.00000186007`. The 95% source-cluster bootstrap interval for step 0 minus
step 25 is `[-0.0000121873, +0.00000720560]`, which includes zero. These two
boundaries are statistically tied, but step 25 does not have a better point
estimate for the declared selection metric. No nonzero optimizer boundary has
a lower mean KLD than step 0.

The wider 10,480,640-position comparison makes the failed generalization of
the screen-selected boundary unambiguous:

| Correction | Mean KLD |
| --- | ---: |
| Weighted-SVD initialization at step 0 | **0.003121866** |
| Gradient-trained correction at step 1300 | 0.003253161 |

The step-1300 minus step-0 difference is `+0.000131294`, or approximately
4.21% worse. Its 95% source-cluster bootstrap interval is
`[+0.000126013, +0.000136936]`, entirely above zero. The subset with no exact
12-token training-corpus overlap gives the same conclusion.

## What the experiment establishes

### Qualified findings

- Direct QSRT K5 encoding and packed execution are correct for all 192 dense
  MLP matrices.
- Activation-weighted SVD provides useful low-rank recovery. Rank 16 is the
  smallest tested rank statistically tied with rank 32 and uses half as much
  adapter storage.
- The selected step-0 correction survives packing: decoded mean KLD is
  `0.003121866`, and the packed B12X kernel/backend stack has mean KLD
  `0.003122880` over 10,480,640 positions.
- On a fixed 32,768-token comparison, QSRT K5 plus rank-16 recovery has mean
  KLD `0.00274573`, versus `0.00633584` for MXFP8-T3.
- The packed checkpoint beats MXFP8-T3 on 831 of 832 clean qualification
  contexts.
- Forty deterministic tasks pass without a BF16 regression. The public
  capability sample passes 56 of 70 questions and retains 55 of the 57
  questions passed by BF16.
- Target-only serving, MTP3, DSpark7, vision, and prefill at 2K, 8K, and 32K
  function under the qualified runtime contract.

### Unsupported claims

- Gradient-based rank-16 adapter QAT improves general distribution fidelity.
- Updating QSRT K5 base weights, RMSNorm gains, or other shared BF16 weights
  would produce the same result. Those parameters were frozen.
- Quantizing attention projections or the language-model head would retain the
  measured quality. Those components remain BF16.
- The artifact is a production replacement for MXFP8. Target-only decode is
  `2.18 tok/s`, versus `20.40 tok/s` for the paired MXFP8 configuration.
- Whole-model Compute Unified Device Architecture (CUDA) graph replay is
  supported. The qualified runtime uses `torch.compile` with CUDA graphs
  disabled.

The result does not show that adapter QAT is universally ineffective. It shows
that one epoch with rank 16, learning rate `2e-5`, a 50-million-token corpus,
and frozen base weights did not improve the broad held-out metric beyond the
weighted-SVD initialization.

## Selected artifact

The selected checkpoint is
`Qwen3.8-27B-QSRT-K5-MLP-r16-quality-step0000-v1`. In this name,
`step0000` means the weighted-SVD correction before any AdamW update; it does
not mean that the correction tensors are zero or randomly initialized.

```text
Checkpoint manifest SHA-256:
fdeb0734392d42ec3e3a69f918faf4e240d9bf17e8dc451307cbee3aadb3d114

Published runtime image:
voipmonitor/vllm@sha256:47109f9fa6d84ad15e3b92615c186d9a9a413dec8ef07f6a6410f96f75a48b6a
```

Research-artifact publication and runtime functionality are qualified. The
checkpoint should be treated as a quality result and an implementation
reference, not as a fast serving tier.

## What a stronger QAT experiment would require

The evidence supports two distinct research directions:

1. Train the decoded dense MLP base weights, with optional RMSNorm or other
   explicitly declared BF16 parameters, rather than training only additive
   rank-16 factors. This is the dense-weight QAT proposal; it was not exercised
   by this run.
2. Preserve separate data roles: training data, a large stratified monitoring
   set, a checkpoint-selection set, and an untouched final test set. A small
   repeatedly observed screen must not decide the published checkpoint.

Any follow-up must compare the weighted-SVD boundary and every trained
candidate on identical token populations. Training-batch loss and screening
curves remain useful diagnostics, but neither is release evidence by itself.

## Related implementation records

The detailed implementation is split across the following pull requests and
the qualification issue:

- [Qualification and implementation issue #78](https://github.com/local-inference-lab/rtx6kpro/issues/78)
- [QSRT implementation pull request #9](https://github.com/local-inference-lab/qsrt/pull/9)
- [Packed Qwen QSRT K5 B12X pull request #236](https://github.com/local-inference-lab/b12x/pull/236)
- [vLLM inference engine missing-vendored-rotary fallback pull request #461](https://github.com/local-inference-lab/vllm/pull/461)
- [vLLM B12X page-table capacity pull request #462](https://github.com/local-inference-lab/vllm/pull/462)
- [Source-locked Docker assembly pull request #27](https://github.com/local-inference-lab/blackwell-llm-docker/pull/27)
