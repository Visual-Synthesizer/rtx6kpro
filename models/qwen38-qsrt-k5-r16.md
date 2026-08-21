# Qwen3.8-27B QSRT K5 dense-MLP recovery

This page specifies and reports the research-only
`Qwen3.8-27B-QSRT-K5-MLP-r16-quality-step0000-v1` checkpoint. Every dense
decoder MLP matrix uses the five-bit Quantile-Stratified Rate-shifted Trellis
(QSRT) format, and every quantized matrix has an additive rank-16 BF16
correction initialized by activation-weighted singular value decomposition
(SVD).

**Evidence snapshot:** 2026-08-21

## Scope and status

| Scope | Status | Result |
| --- | --- | --- |
| QSRT K5 artifact integrity | `qualified` | All 192 dense-MLP matrices across 64 decoder layers have immutable packed payloads and independent decode receipts |
| Weighted-SVD low-rank recovery | `qualified` | Rank 16 is the smallest candidate statistically tied with rank 32 and improves the frozen full-model KLD point estimate over raw K5 and a matched-byte partial-K6 control |
| Gradient-based adapter recovery | `unsupported` | No nonzero optimizer boundary beats weighted-SVD initialization on the balanced held-out population |
| Packed dense numerical execution | `qualified` | All 192 payloads pass output, input-Jacobian, CUDA-graph, immutability, buffer, and allocation gates |
| Whole-model CUDA-graph replay | `unsupported` | Controlled retrieval scores 4/8 with graphs and 8/8 in eager and compiled no-graph modes; the guarded runtime rejects graph mode before weight loading |
| 32,768-token distribution fidelity | `qualified` | QSRT K5 plus rank-16 recovery beats the paired MXFP8-T3 checkpoint on mean KLD, p99 KLD, and top-1 agreement |
| Full-population distribution fidelity | `qualified` | All 10,480,640 positions are scored; packed execution preserves decoded quality and beats MXFP8 on the clean qualification population |
| Deterministic task and long-context retention | `qualified` | 40/40 tasks pass with no BF16 regression; all eight needles are retrieved at approximately 2k, 100k, and 195k context levels |
| Public capability retention | `qualified` | 56/70 questions pass and 55/57 BF16-pass questions remain passing; the paired run has two regressions and one improvement |
| Serving functionality | `qualified` | Target-only, MTP3, DSpark7, 2K/8K/32K prefill, and the fixed vision fixture pass |
| Serving performance | `unsupported` | QSRT target-only decode is 10.67% of the paired MXFP8-T3 median |
| Production MXFP8-tier replacement | `unsupported` | Attention-path tensors remain BF16 and production serving performance is not qualified |

The selected optimizer boundary is step 0. Step 0 contains the weighted-SVD
initialization before any AdamW update. The artifact therefore demonstrates
low-rank recovery of quantization error; it does not demonstrate a quality
improvement caused by gradient-based quantization-aware training (QAT).

## Artifact contract

The official BF16 source is
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The text model
contains 64 decoder layers, including 48 gated-DeltaNet layers and 16
full-attention layers. Each layer has separate dense `gate_proj`, `up_proj`,
and `down_proj` matrices.

The quantized scope contains 17,112,760,320 coefficients in 192 matrices.
Direct Viterbi encoding with the QSRT `sqg_fp16_d3l` reconstruction law emits
10,695,475,200 trellis bytes at exactly 5.0 encoded bits per coefficient and
8,650,752 FP16 scale bytes. The logical K5 payload is 10,704,125,952 bytes
before container overhead.

Each matrix executes the additive correction

```text
y = decode(W_QSRT_K5) x + B (A.T x)
```

where `A` and `B` are BF16 rank-16 factors. The checkpoint contains 384 factor
tensors, 69,206,016 factor parameters, and 138,412,032 logical BF16 factor
bytes. The K5 payload, scales, transforms, and every retained BF16 tensor are
frozen.

Full-attention projections, gated-DeltaNet projections, RMSNorm parameters,
embeddings, the vision tower, the multi-token prediction head, and `lm_head`
remain BF16. The checkpoint does not contain MCG or EXL3 payloads and is not a
transcode of an MXFP8 checkpoint.

Immutable checkpoint identities:

- packed checkpoint directory:
  `/root/qwen38/Qwen3.8-27B-QSRT-K5-MLP-r16-quality-step0000-v1`;
- packed-checkpoint manifest SHA-256:
  `fdeb0734392d42ec3e3a69f918faf4e240d9bf17e8dc451307cbee3aadb3d114`;
- selected overlay SHA-256:
  `99f478c7dfa17e16cf88442e7c8d6b51250d068a8e9392ed8be106346002c8ae`;
- all-boundary selection-report SHA-256:
  `a7e9392e10abf5998fd1855676e0c921d90ca9f4f577b70140b409d4bfc43d97`.

## SVD rank analysis

The activation archive contains 24,576 fit rows from 512 complete documents
and 8,192 validation rows from another 256 documents at each of 128 distinct
dense-MLP input interfaces. Its manifest SHA-256 is
`4b9892e9da44f2bc136389e954d06a3ea9ce6c63feb70bba89f83dba4ada97f7`.

Activation-weighted rank 16 captures 25.0279501% of held-out operator-output
error. Plain rank 16 captures 1.3742785%. A frozen full-model screen over
65,480 causal positions gives:

| Arm | Mean KLD | p99 KLD | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Raw decoded K5 anchor | 0.01588853 | 0.173095 | 97.1274% |
| Weighted rank 8 | 0.01519351 | 0.130520 | 97.3564% |
| Weighted rank 16 | **0.01479363** | 0.119908 | 97.3870% |
| Weighted rank 32 | 0.01507232 | **0.116607** | **97.3977%** |

The paired rank-16 versus rank-32 intervals include zero for mean KLD, p99
KLD, and top-1 agreement. Rank 16 is the smallest statistically tied choice
and uses half the factor storage. A matched-byte partial-K6 control has mean
KLD 0.01561512 and p99 KLD 0.168765. Rank-16 recovery has better point
estimates for mean, p99, and top-1; the p99 and top-1 advantages are resolved,
while the mean-KLD interval includes zero.

## Recovery training and checkpoint selection

The recovery corpus contains 35,790 complete documents from 11 sources and
exactly 50,000,000 stored tokens. Independent replay with the pinned tokenizer
reproduces the token count without a document mismatch. The one-epoch run
processed 49,964,210 causal positions in 1,680 optimizer groups.

AdamW used learning rate `2e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, 100
warmup groups, zero ordinary weight decay, BF16 compute, and FP32 masters,
moments, and accumulated gradients. Only the 384 factor tensors were
trainable. Every K5 base and every retained BF16 tensor remained frozen.

Two pipeline owners used two RTX PRO 6000 GPUs for optimizer groups 0 through
399. Two synchronous replicas of that two-owner pipeline used four GPUs for
groups 400 through 1,679. Wall time was 104,937.106 seconds (29.15 hours),
throughput was 476.135 causal positions/s, and measured pipeline bubble
fraction was 0.3331.

All 69 durable boundaries—step 0, every 25 optimizer groups, and the epoch
boundary—were scored on a balanced analysis-only population containing 512
contexts, 1,048,064 causal positions, five equal-weight strata, and 366 source
clusters. No context has an exact 12-token match with the recovery corpus.

| Optimizer boundary | Mean KLD | p99 KLD | Top-1 agreement |
| ---: | ---: | ---: | ---: |
| 0 | **0.00316779494** | **0.0319588** | 97.4637% |
| 25 | 0.00316965500 | 0.0320798 | **97.4726%** |
| 50 | 0.00317031509 | 0.0320911 | 97.4703% |
| 100 | 0.00319279973 | 0.0321952 | 97.4565% |
| 1300 | 0.00330893583 | 0.0331666 | 97.4247% |

Step 0 minus step 25 is `-0.00000186007` mean KLD, with a 95%
source-cluster bootstrap interval of
`[-0.0000121873, +0.00000720560]`. The two boundaries are statistically tied,
and no nonzero boundary improves the registered selection metric. A
56-document screening archive is source-concentrated and is unsupported as a
checkpoint-selection population.

This run is adapter-only full-model QAT: full-depth autograd minimizes
forward KLD from token IDs through all 64 layers, but the dense base weights
remain frozen. QAT that updates decoded dense base weights, RMSNorm gains, or
other retained BF16 tensors is a separate experiment.

## Numerical and distribution evidence

The packed B12X operator qualification covers all 64 layers and all 192 K5
payloads. Every layer runs one-row and 128-row cases. Layers 0, 31, and 63
also run 8, 32, 512, 2,048, and 8,192 rows. Every case passes:

- output and input-vector Jacobian parity against decoded BF16;
- immutable packed weights and stable caller-owned buffer pointers;
- no replay-time allocation;
- bit-exact eager and CUDA-graph replay in deterministic correctness mode;
- declared workspace capacity through 8,192 rows.

The 32,768-token top-4096 comparison uses one sealed token sequence, the same
runtime image, and 32,767 scored positions. Shared BF16 probability mass is
above 99.8%.

| Checkpoint | Mean KLD | p99 KLD | Top-1 agreement |
| --- | ---: | ---: | ---: |
| MXFP8-T3 | 0.00633584211 | 0.0492137 | 96.9451% |
| QSRT K5 plus rank-16 recovery | **0.00274572946** | **0.0229173** | **97.7020%** |

The wider 5,120-context fidelity population contains 10,480,640 scored
positions. Body-only measurements project every checkpoint through one shared
BF16 output head. They measure text-body error and do not measure a serialized
checkpoint's output-head quantization. KLD numbers from different token
populations are not directly comparable.

| Body and runtime-native BF16 reference | Mean KLD | p99 KLD | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Hydrated EXL3 K5/K6, vLLM BF16 | **0.00275962502** | not published | **97.7009%** |
| Decoded QSRT K5 plus weighted rank 16, Hugging Face BF16 | 0.00312186620 | **0.0312664** | 97.5056% |
| Packed QSRT K5 plus weighted rank 16, vLLM BF16 | 0.00312287968 | 0.0313921 | 97.5076% |
| Online-overlay EXL3 K5/K6, vLLM BF16 | 0.00320988396 | not published | 97.5180% |
| Static FP8 control, vLLM BF16 | 0.00529394126 | not published | 96.7944% |

The contamination-excluded qualification comparison retains 832 contexts and
1,703,104 scored positions after removing 224 contexts with any exact
12-token recovery-corpus match. Packed B12X has mean KLD `0.00316525573`
versus decoded-oracle `0.00314226878`, a ratio of `1.0073154` within the
declared `1.05` preservation limit. Its top-1 agreement trails by
`0.00003934`.

Packed QSRT beats MXFP8-T3 by mean KLD `-0.00312210337`, with a 95%
source-cluster interval of
`[-0.00380504174, -0.00264499683]`, a negative equal-stratum difference, and
831/832 context wins. Hydrated EXL3 beats packed QSRT by `0.00044548394`, with
interval `[0.00038247271, 0.00051861148]`. Packed QSRT and online-overlay EXL3
are unresolved: packed-minus-online is `+0.00000403270`, with interval
`[-0.0000433562, +0.0000591751]`. Packed QSRT beats static FP8 by
`-0.00210079900`, with an entirely negative interval.

The fixed public-capability population contains 70 questions evaluated with
identical prompts and scoring rules for packed QSRT and BF16. Packed QSRT
passes 56/70 questions, retains 55 of the 57 questions passed by BF16, has two
regressions and one improvement, and agrees with BF16 on 67/70 pass/fail
outcomes. The public-capability gate is qualified. Exact generated-text
identity is 0/70 because the numerical model perturbation changes generation;
exact text identity is reported as diagnostic evidence and is not the declared
capability-retention gate. The public report SHA-256 is
`e21901e6708dd7e4124ba201bea7a50307983d054e5b5ec4be9fe27d8574ad6b`.
The task, retrieval, and public-capability aggregate SHA-256 is
`7dfb7790f52d3d568fa33353798e067692414b4cf563f9fa45aad80d13453151`.

## Source-locked runtime image

The local qualification image is:

```text
voipmonitor/vllm:qwen38-qsrt-k5-r16-vllm1882179-b12x28f1007-cu133-torch213-20260821-r7
sha256:48650edc867dcd400f13886ba9e2bd6a429efa4f255b3a63ff5a82df9788fd93
```

Its size is 30,882,322,729 bytes. It uses CUDA 13.3, PyTorch 2.13.0, vLLM
combined source tree `188217905a37f04ce50659441a03fa4f7256435f`, B12X
combined source tree `28f10076d5df9898d0d68ac41edec0f787c93b57`, and QSRT
packager revision `3fb7b0a1f020d2667d93e356ee92639fa34abfff`.

The image derives from immutable base manifest
`voipmonitor/vllm@sha256:ff9d4f2402ed88b1ae7ca3a6886c80a64d72993f1a593380c8cb6f193437567d`
and base image ID
`sha256:b8ce67bd8ed86ad9a77affe63105b1ace4f7a6a8e09b41e1ba5deb9379a3e81e`.
The Qwen overlay recipe is commit
`ffcc350a9e6d5efc4e5727a641264556f12fd474` in
[`local-inference-lab/blackwell-llm-docker#27`](https://github.com/local-inference-lab/blackwell-llm-docker/pull/27).

Build-time validation and the pinned-image GPU preflight pass 93 tests plus
exact checks for EXL3 Hadamard compatibility, the missing-vendored-rotary
fallback, a compiled full-graph trace of the direct K5-plus-rank-16 operator,
the whole-model CUDA-graph guard, and hybrid page-table capacity. The default
launcher retains torch.compile, disables CUDA graphs, and retrieves all eight
deterministic 2K-context needles at `max_model_len=261632`.

The published registry identity is:

```text
voipmonitor/vllm@sha256:47109f9fa6d84ad15e3b92615c186d9a9a413dec8ef07f6a6410f96f75a48b6a
```

The local image ID identifies the qualified build; the registry digest above
is the immutable pull identity.

## Release disposition

Research-artifact publication is `qualified`. The scope-specific release
decision is bound by SHA-256
`52d4fea4cad5ef112c5d4b616b0d1138cbc2d1b8ea95c96aff064803778999bd`.
Its image-equivalence receipt has SHA-256
`ef0fe082520e5fa8b15d6db1fdcb355d7d9a52a781c8b28ee29371028b17d996`,
and its registry-publication receipt has SHA-256
`c970a3df90068fefa3e3349a36d5f4e5ff45c92262b9b92c8264fbc6b3b84754`.

The release decision qualifies packed-checkpoint quality, research-artifact
publication, and runtime functionality. Gradient-based adapter-QAT efficacy,
whole-model CUDA graphs, serving performance, and production-tier replacement
remain `unsupported`. The publication does not reclassify those scopes.

## Pull-request inventory

The image retains exact commits even when a pull-request branch later moves.
“Fork-specific” means that the pull-request body declares no direct
`vllm-project/vllm` equivalent. Closed-unmerged fork PRs remain in the image
because the base integration tree was qualified with those exact revisions;
their state does not rewrite the immutable image contents.

### vLLM

| PR | Image pin | Fork state at 2026-08-21 | Upstream relationship |
| --- | --- | --- | --- |
| [#285](https://github.com/local-inference-lab/vllm/pull/285) resolved revision identity | `d6b7377cbf27` | open | Fork-specific; no upstream mapping declared |
| [#286](https://github.com/local-inference-lab/vllm/pull/286) in-checkpoint draft identity | `9dbec23e2b2a` | open | Fork-specific; no upstream mapping declared |
| [#287](https://github.com/local-inference-lab/vllm/pull/287) DS4 launcher identity and capacity | `a5e76352ba29` | open | Fork-specific; no upstream mapping declared |
| [#288](https://github.com/local-inference-lab/vllm/pull/288) B12X mHC input contract | `29f13ebc717d` | open | Fork-specific; no upstream mapping declared |
| [#289](https://github.com/local-inference-lab/vllm/pull/289) sparse metadata active width | `91fd9f8f74de` | closed unmerged | Fork-specific; no upstream mapping declared |
| [#290](https://github.com/local-inference-lab/vllm/pull/290) serving-shape memory profile | `c13a214bd703` | open | Fork-specific; no upstream mapping declared |
| [#292](https://github.com/local-inference-lab/vllm/pull/292) canonical B12X identifiers | `fb8d983acd64` | open | Fork-specific; no upstream mapping declared |
| [#293](https://github.com/local-inference-lab/vllm/pull/293) hybrid KV load recovery | `0e41faa811ce` | closed unmerged | Fork-specific; no upstream mapping declared |
| [#294](https://github.com/local-inference-lab/vllm/pull/294) grammar bitmask source width | `826bb4088f46` | open; head `d931e0d45dc9` | [upstream #52436](https://github.com/vllm-project/vllm/pull/52436), merged |
| [#295](https://github.com/local-inference-lab/vllm/pull/295) termination-safe XGrammar batches | `71b93dc6f3d5` | open; head `b115455fd0f3` | [upstream #52805](https://github.com/vllm-project/vllm/pull/52805), merged |
| [#296](https://github.com/local-inference-lab/vllm/pull/296) duplicate DSML closer suppression | `31472fd994f9` | open | Fork-specific; no upstream mapping declared |
| [#298](https://github.com/local-inference-lab/vllm/pull/298) FULL-graph decode-state gate | `b8d92cb5d26e` | open | [upstream #51865](https://github.com/vllm-project/vllm/pull/51865), merged |
| [#300](https://github.com/local-inference-lab/vllm/pull/300) projection-mixed EXL3 experts | `901a7c50e5d3` | open; head `1b0c1c49f351` | Fork-specific; no upstream mapping declared |
| [#301](https://github.com/local-inference-lab/vllm/pull/301) GLM-5.2 sparse-MLA contracts | `2255f632485c` | open | Fork-specific; no upstream mapping declared |
| [#302](https://github.com/local-inference-lab/vllm/pull/302) reasoning-aware strict-tool grammar | `7d1c21353cf4` | open | Fork-specific; no upstream mapping declared |
| [#303](https://github.com/local-inference-lab/vllm/pull/303) DeepSeek-V4 draft quantization | `4b297d1a07bf` | open | [upstream #51835](https://github.com/vllm-project/vllm/pull/51835), closed unmerged |
| [#304](https://github.com/local-inference-lab/vllm/pull/304) create-if-absent filesystem KV publication | `229de6270e51` | open | Fork-specific; no upstream mapping declared |
| [#308](https://github.com/local-inference-lab/vllm/pull/308) heterogeneous KV block zeroing | `053e6351d0b3` | open | [upstream #51749](https://github.com/vllm-project/vllm/pull/51749) and [#52058](https://github.com/vllm-project/vllm/pull/52058), merged |
| [#309](https://github.com/local-inference-lab/vllm/pull/309) deferred MLA DCP workspace | `dc0c026df624` | open | Fork-specific; no upstream mapping declared |
| [#320](https://github.com/local-inference-lab/vllm/pull/320) speculative structured-output validation | `e9534672129b` | open; head `fd0237e15f71` | [upstream #52452](https://github.com/vllm-project/vllm/pull/52452), open with merge conflicts |
| [#415](https://github.com/local-inference-lab/vllm/pull/415) DSpark CUDA-graph capture contract | `c805ebd0896c` | open; head `2e8535c70af8` | Fork-specific; no upstream mapping declared |
| [#417](https://github.com/local-inference-lab/vllm/pull/417) legacy direct DSML tool calls | `2511e5df2b1e` | open | Fork-specific; no upstream mapping declared |
| [#461](https://github.com/local-inference-lab/vllm/pull/461) missing-vendored-rotary fallback | `0942707892ca` | open; ready for review; head `a434f31d77ed` | Distinct from [upstream #52121](https://github.com/vllm-project/vllm/pull/52121), open and blocked, which handles an installed but unloadable dependency |
| [#462](https://github.com/local-inference-lab/vllm/pull/462) B12X hybrid page-table capacity | `1d341d8482e4` | open; ready for review | Fork-specific B12X integration; no upstream mapping |

### B12X

| PR | Image pin | Fork state at 2026-08-21 | Upstream relationship |
| --- | --- | --- | --- |
| [#145](https://github.com/local-inference-lab/b12x/pull/145) CUTLASS DSL 4.6.2 | `7f88972df71d` | closed unmerged | Fork-specific; no upstream mapping declared |
| [#221](https://github.com/local-inference-lab/b12x/pull/221) unpaired K6/MCG dense kernel | `413f96e889da` | open | Fork-specific; no upstream mapping declared |
| [#223](https://github.com/local-inference-lab/b12x/pull/223) projection-mixed routed experts | `e99775f552c4` | open; head `3df80ee36e2a` | Fork-specific; no upstream mapping declared |
| [#227](https://github.com/local-inference-lab/b12x/pull/227) inactive native W4A16 routes | `e38436d76a95` | open; head `0eba6ae99e0d` | Fork-specific; no upstream mapping declared |
| [#228](https://github.com/local-inference-lab/b12x/pull/228) inactive tiny-decode routes | `50046df84a15` | merged | Fork-specific; no upstream mapping declared |
| [#229](https://github.com/local-inference-lab/b12x/pull/229) CUDA-graph buffer tests | `2cdd9e265cd6` | merged | Fork-specific; no upstream mapping declared |
| [#230](https://github.com/local-inference-lab/b12x/pull/230) mapped-route namespace planning | `156920046e85` | merged | Fork-specific; no upstream mapping declared |
| [#236](https://github.com/local-inference-lab/b12x/pull/236) packed Qwen QSRT K5 dense inference | `1dfe87039951` | open; ready for review; head `fe96602864fe`; all 11 review threads resolved; bot re-review rate-limited | Fork-specific; no upstream mapping declared |

The Qwen image applies a content-checked reconciliation patch after B12X
#236. Its SHA-256 is
`249a29c47efb3a1637dbe9dd0923de38f4cbeaee1573dea9a88a00409568a0c3`.
The patch preserves previously qualified W4A16 and K6/MCG changes in the
three overlapping B12X files and locks CUTLASS DSL 4.6.2.

### LMCache inherited from the base image

LMCache is present in the source-locked base image but is disabled by the
Qwen launcher. The composed LMCache tree is
`e045d729bc5c4c63a40e13d032f42923de97812f` on base commit
`a128b2e286ebb3556cb43124149e600ff99fe481`.

| PR | Image pin | Fork state at 2026-08-21 | Upstream relationship |
| --- | --- | --- | --- |
| [#7](https://github.com/local-inference-lab/LMCache/pull/7) MP handler failures | `7b7583aef55e` | merged | Fork release branch; no upstream mapping declared |
| [#8](https://github.com/local-inference-lab/LMCache/pull/8) safe MP retrieval recompute | `31c4175d2134` | merged | Fork release branch; no upstream mapping declared |
| [#9](https://github.com/local-inference-lab/LMCache/pull/9) largest-prefix L1 preservation | `c3b73de74d85` | merged | Fork release branch; no upstream mapping declared |
| [#10](https://github.com/local-inference-lab/LMCache/pull/10) active-lookup eviction guard | `59186798bb58` | merged | Fork release branch; no upstream mapping declared |
| [#11](https://github.com/local-inference-lab/LMCache/pull/11) bounded failed-key telemetry | `2cee5a2c1ef7` | merged | Fork release branch; no upstream mapping declared |
| [#12](https://github.com/local-inference-lab/LMCache/pull/12) per-key native SET results | `baf1345106eb` | merged | Fork release branch; no upstream mapping declared |
| [#13](https://github.com/local-inference-lab/LMCache/pull/13) O_DIRECT alignment | `d4dce69bf903` | merged | Fork release branch; no upstream mapping declared |
| [#14](https://github.com/local-inference-lab/LMCache/pull/14) object-group filesystem keys | `502b3c60f2e6` | merged | Fork release branch; no upstream mapping declared |
| [#15](https://github.com/local-inference-lab/LMCache/pull/15) durable bounded native stores | `7729a9c75f2c` | merged | Fork release branch; no upstream mapping declared |
| [#16](https://github.com/local-inference-lab/LMCache/pull/16) restart accounting recovery | `e6c2708fe8ae` | merged | Fork release branch; no upstream mapping declared |
| [#17](https://github.com/local-inference-lab/LMCache/pull/17) durable L1 writeback | `f9639e2094f7` | merged | Fork release branch; no upstream mapping declared |
| [#22](https://github.com/local-inference-lab/LMCache/pull/22) hybrid object-group prefetch | `084e0797f03d` | merged | Fork release branch; no upstream mapping declared |
| [#23](https://github.com/local-inference-lab/LMCache/pull/23) writer-owned filesystem publication | `1bbe09a1a183` | merged | Fork release branch; no upstream mapping declared |

QSRT pull request
[#9](https://github.com/local-inference-lab/qsrt/pull/9) at
`3fb7b0a1f020d2667d93e356ee92639fa34abfff` defines the artifact packager,
all-boundary selection contract, and content-bound recovery receipts. It is an
offline artifact identity rather than an inherited base-image patch.

vLLM pull request
[#427](https://github.com/local-inference-lab/vllm/pull/427) is intentionally
excluded. It changes Kimi-only interleaved MLA query materialization and does
not change Qwen execution.

## Serving contract and limitations

The image entrypoint serves one GPU with `--linear-backend b12x`,
`--attention-backend B12X_ATTN`, block size 128, disabled prefix caching, and
`B12X_QSRT_MODEL_DIR` bound to the materialized checkpoint. It retains
torch.compile with `cudagraph_mode=NONE`. vLLM #462 sizes
the B12X scratch page table for the model's hybrid-cache property. At
`max_model_len=65536`, an 896-token storage alignment needs 518 page-table
columns rather than the 512 columns implied by a pure 128-token-page model.

Direct packed-projection CUDA-graph capture remains qualified. Whole-model
vLLM CUDA-graph replay is unsupported. The same deterministic 2K retrieval
population scores 4/8 under graph replay and 8/8 in eager and compiled no-graph
modes. B12X #236 rejects graph-enabled whole-model execution before checkpoint
weight loading.

Functional target-only, MTP3, DSpark7, 2K/8K/32K prefill, and vision checks
are distinct from performance qualification. Five sequential decode samples
and three prefill samples detect material regressions but are not a production
load test.

| Decode mode | Packed QSRT median | MXFP8-T3 median | QSRT acceptance | MXFP8 acceptance |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 2.1763 tok/s | 20.4046 tok/s | not applicable | not applicable |
| MTP3 | 8.2802 tok/s | 62.6089 tok/s | 95.96% | 90.10% |
| DSpark7 | 11.8863 tok/s | 94.2202 tok/s | 65.22% | 63.53% |

| Prompt length | Packed QSRT prefill | MXFP8-T3 prefill |
| ---: | ---: | ---: |
| 2,048 | 1,208.20 tok/s | 8,745.57 tok/s |
| 8,192 | 1,425.14 tok/s | 8,032.39 tok/s |
| 32,768 | 1,392.76 tok/s | 7,023.92 tok/s |

All six decode arms complete five fixed 512-token outputs, both speculative
arms publish nonzero counters, and every prefill arm completes three samples.
The retained BF16 vision path identifies the fixed fixture as red on the left
and blue on the right. Runtime functionality is qualified. Serving performance
is unsupported because packed target-only decode is 10.67% of the MXFP8-T3
median, below the declared 95% gate. The serving aggregate SHA-256 is
`21cb4b527249517ce71b7a497a8f488c7826ccbb356a841a93b9345bb7e9761c`.

Real QAT over dense base weights, RMSNorm/shared-weight training,
attention-path K5/K6, output-head quantization, and production concurrency
qualification are unsupported by this artifact. Each requires a separate
trainable inventory, optimizer-memory smoke, corpus identity, checkpoint
identity, and held-out evaluation.
