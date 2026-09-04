# GLM-5.3-Flash BF16-to-NVFP4 distribution fidelity

Status: **research-only**.

This report measures how the next-token distribution of the pinned
GLM-5.3-Flash NVIDIA 4-bit floating-point (NVFP4) checkpoint differs from the
pinned Brain Floating Point 16-bit (BF16) checkpoint. It also uses
route-controlled Mixture-of-Experts (MoE) executions to distinguish
distribution changes observed with a shared route sequence from changes caused
by feedback between quantized hidden states and later router decisions.

The primary natural-route result is `0.176065` nats/token forward
Kullback-Leibler Divergence (KLD) on 1,571,435 development-facing positions and
`0.161911` nats/token on 524,020 held-out positions. When both checkpoints
execute an identical captured route sequence, forward divergence is
approximately `0.067` to `0.071` nats/token.

No quality threshold is defined. The route-controlled values are
counterfactual measurements, not additive components of the natural-route
result. None of these measurements establishes by itself whether the NVFP4
checkpoint is acceptable for a workload.

## Measurement target

| Operand | Checkpoint identity |
|---|---|
| Reference | `zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a` |
| Candidate | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |

Inference used Tensor Parallelism 8 (TP8), eager execution, disabled Torch
compilation, NVIDIA Collective Communications Library (NCCL) tensor-parallel
reduction, and sequential shared-expert execution. The NVFP4 captures use the
reproducible `FLASHINFER_CUTLASS` MoE
backend in instrumented runtime image `local/glm53-kld-r18:dev-20260903`. The
capture-host image ID is
`sha256:54c92a17da2f8cdfdb75440bf58e535b35ef0980d547c52226f57e91ae0b6d40`.

The B12X NVFP4 dynamic MoE backend is **unsupported for this evaluation**.
Repeated calls to its real layer-3 kernel with fixed inputs, expert IDs, route
weights, fresh bindings, and cleared caller-owned buffers differ in 3,004,188
of 8,388,608 BF16 output elements. Maximum absolute error is `0.2915`, and mean
absolute error is `0.01375`. The defect is inside the B12X NVFP4 dynamic MoE
kernel rather than tokenization, request state, route capture, shared-expert
execution, or stale caller-owned buffers.

Consequently, the reported KLD characterizes the pinned NVFP4 checkpoint
executed by `FLASHINFER_CUTLASS`; it does not characterize the qualified
Jovian Judgement serving configuration, which uses B12X NVFP4 MoE.

All captured final hidden states are projected through one shared BF16
language-model head. Its safetensors file has SHA-256
`ace852f317fb56d240f539e4dbffc0883827cfc1f58b3df5ba2d389289801869`.
This controls the output projection and attributes measured differences to the
transformer hidden states rather than to independently rounded output heads.

## Evaluation suite

The GLM evaluation suite has durable identifier
`glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1`. It inherits source
selection and source-cluster assignments from the
[`festr2/kimi-k3-distribution-fidelity-1024x2048-v1`](https://huggingface.co/datasets/festr2/kimi-k3-distribution-fidelity-1024x2048-v1)
artifact at revision `402919ae70d61396087571b63fe9185d95491afb`, but it
contains GLM-specific token artifacts. Text is tokenized with the pinned GLM
BF16 tokenizer; dialogue data uses the model's native chat template. Kimi-K3
hidden states are not used as a GLM reference.

The suite contains 1,024 contexts with at most 2,048 prompt tokens. Its durable
partition identifiers are:

- `analysis`: 768 development-facing contexts and 1,571,435 scored positions.
- `qualification`: 256 held-out contexts and 524,020 scored positions.

Source clusters do not cross the partition boundary. The suite manifest has
SHA-256
`50520bdba81a9447b769e72da58720fb8468bb6dd19f641493c9110b04f9972b`,
and the ordered suite token hash is
`e2c541bce4a3213f697cd5236eaa60393d044dc054ea3ec2cc35d79dcb089f9b`.

The held-out metric computation was specified before its distribution metrics
were computed or inspected. The frozen evaluation contract has SHA-256
`b19dcc0b4b155bdb60dcb32c1aab54548d3ab0fdcd83be01b03feff3da76c07f`.
It defines a descriptive evaluation and no pass/fail cutoff.

## Route-controlled design

A route sequence contains the selected top-eight expert IDs and associated
router weights for every prompt token and every MoE layer from layer 3 through
layer 44. Route replay overrides both values at the logical point after top-k
selection and before Expert Parallel Load Balancing (EPLB) remapping.

| Checkpoint | Route sequence | Semantic role |
|---|---|---|
| BF16 | Derived by the same BF16 execution | Natural reference |
| NVFP4 | Derived by the same NVFP4 execution | Natural candidate |
| NVFP4 | Captured from the BF16 natural execution | Candidate under the reference route path |
| BF16 | Captured from the NVFP4 natural execution | Reference under the candidate route path |

The four cells support five directional comparisons:

1. Natural end-to-end divergence:
   `KL(BF16 with BF16 routes || NVFP4 with NVFP4 routes)`.
2. Checkpoint divergence conditional on BF16 routes:
   `KL(BF16 with BF16 routes || NVFP4 with BF16 routes)`.
3. Checkpoint divergence conditional on NVFP4 routes:
   `KL(BF16 with NVFP4 routes || NVFP4 with NVFP4 routes)`.
4. BF16 sensitivity to the NVFP4 route sequence:
   `KL(BF16 with BF16 routes || BF16 with NVFP4 routes)`.
5. NVFP4 sensitivity to the BF16 route sequence:
   `KL(NVFP4 with NVFP4 routes || NVFP4 with BF16 routes)`.

Forward KLD is asymmetric. The route-sensitivity comparisons do not estimate
interchangeable distances, and none of the route-controlled values may be
subtracted from or summed into an exact decomposition of natural-route KLD.
The [general vLLM KLD protocol](README.md) explains the estimands, controls,
statistics, and route-trace contract.

## Distribution computation

For each scored position, the comparator reconstructs all 154,880 logits from
the captured hidden state and the shared BF16 language-model head. It evaluates

```text
D_KL(P || Q) = sum_i P_i * (log(P_i) - log(Q_i))
```

over the complete vocabulary. No top-k or top-p truncation is applied. The
comparator, with SHA-256
`54b1ab1c24ca5d671c00fcd27f693f8a489024a55521b34f8bfeb12d27f20b7b`,
uses a full-vocabulary row-maximum pass, 32-bit floating-point shifted
exponentials, 64-bit floating-point probability-mass accumulation, and 64-bit
floating-point weighted reductions. It also reports
Jensen-Shannon divergence and next-token top-1 agreement.

The primary point estimate is the token-weighted arithmetic mean of per-token
forward KLD. The separate uncertainty estimate resamples independent
`(dataset, source_cluster_id)` units 10,000 times with seed 1. The tables report
the source-cluster bootstrap estimate and its percentile 95% interval alongside
the token-weighted point estimate; their weighting differs slightly.

## Distribution results

### Development-facing partition (`analysis`)

| Directional comparison | Token-weighted KLD (nats/token) | Source-cluster estimate (95% CI) | Mean JS | Top-1 agreement |
|---|---:|---:|---:|---:|
| Natural BF16 to natural NVFP4 | `0.176065` | `0.176187` (`0.150526`–`0.206277`) | `0.024733` | 90.485% |
| BF16 to NVFP4, both using BF16 routes | `0.070211` | `0.070223` (`0.063104`–`0.078087`) | `0.013615` | 93.032% |
| BF16 to NVFP4, both using NVFP4 routes | `0.070544` | `0.070560` (`0.063389`–`0.078406`) | `0.013653` | 93.013% |
| BF16 route sensitivity | `0.120246` | `0.120368` (`0.097309`–`0.147432`) | `0.014749` | 93.333% |
| NVFP4 route sensitivity | `0.162206` | `0.162317` (`0.138660`–`0.190207`) | `0.022458` | 90.876% |

### Held-out partition (`qualification`)

| Directional comparison | Token-weighted KLD (nats/token) | Source-cluster estimate (95% CI) | Mean JS | Top-1 agreement |
|---|---:|---:|---:|---:|
| Natural BF16 to natural NVFP4 | `0.161911` | `0.161910` (`0.129389`–`0.202547`) | `0.023774` | 90.680% |
| BF16 to NVFP4, both using BF16 routes | `0.066807` | `0.066806` (`0.058400`–`0.076769`) | `0.013344` | 93.073% |
| BF16 to NVFP4, both using NVFP4 routes | `0.067962` | `0.067962` (`0.058985`–`0.078660`) | `0.013339` | 93.084% |
| BF16 route sensitivity | `0.107732` | `0.107731` (`0.078697`–`0.143865`) | `0.014059` | 93.421% |
| NVFP4 route sensitivity | `0.150156` | `0.150155` (`0.119675`–`0.187748`) | `0.021707` | 90.989% |

The held-out partition preserves the ordering and scale of all five
comparisons. Every held-out point estimate falls inside the corresponding
development-facing source-cluster bootstrap interval. This supports the claim
that the pattern is not confined to development-facing contexts; it is not a
formal equivalence test between partitions.

No comparison produced a negative KLD or Jensen-Shannon value before the
configured roundoff clamp.

## Natural-route agreement

The route comparator, with SHA-256
`85ad17d50f0f0b81f68a1f97061c70bc2bf8236085097857621ffcfff0282879`,
compares natural BF16 and NVFP4 router outputs at every scored token and all 42
MoE layers. Route-weight Total Variation (TV) is computed after independently
normalizing each top-eight weight vector.

| Partition | Token-layer cases | Top-1 expert agreement | Ordered slot agreement | Mean top-8 set overlap | Exact top-8 set agreement | Mean weight TV |
|---|---:|---:|---:|---:|---:|---:|
| `analysis` | 66,000,270 | 89.376% | 55.643% | 87.278% (6.982/8 experts) | 36.919% | `0.112285` |
| `qualification` | 22,008,840 | 89.306% | 55.644% | 87.195% (6.976/8 experts) | 36.917% | `0.113079` |

Layer 3 router outputs are identical because both checkpoints receive the same
embedding and dense-layer prefix before the first MoE block. Divergence appears
after the first quantized MoE computation and propagates through later router
decisions:

| MoE layer | Top-1 agreement | Mean top-8 set overlap | Exact set agreement | Mean weight TV |
|---:|---:|---:|---:|---:|
| 3 | 100.000% | 100.000% | 100.000% | `0.000000` |
| 4 | 98.709% | 97.309% | 78.980% | `0.021009` |
| 5 | 97.275% | 95.148% | 63.970% | `0.039658` |
| 20 | 86.494% | 87.008% | 31.441% | `0.124083` |
| 30 | 85.771% | 83.283% | 28.606% | `0.148567` |
| 40 | 84.690% | 81.472% | 20.770% | `0.163165` |
| 44 | 88.688% | 85.882% | 30.251% | `0.124876` |

An identity test over one 2,047-position context, 42 layers, and 85,974
token-layer cases reports exact agreement for every expert metric and zero
route-weight TV. Source-file hash verification was enabled. The identity
receipt has SHA-256
`6c91a85d4495c7466ebddc9f1cd89d2b028773da39a7a850df39f7021423ce0c`.

## Allocation-stratum result

Natural-route divergence is heterogeneous across the suite's semantic
allocation strata. The table reports source-cluster bootstrap point estimates
in nats/token.

| Allocation stratum | `analysis` | `qualification` |
|---|---:|---:|
| Chinese | `0.116038` | `0.136309` |
| Code, tests, documentation, and issues | `0.056837` | `0.057098` |
| Dialogue, instruction, and assistance | `0.871562` | `0.754141` |
| Encyclopedic and factual | `0.127632` | `0.135579` |
| Literary, narrative, and creative | `0.062886` | `0.053772` |
| News, history, economics, legal, and essays | `0.050260` | `0.058441` |
| Other multilingual | `0.070790` | `0.075453` |
| Scientific and technical | `0.032762` | `0.031122` |
| Structured data, tools, APIs, and tables | `0.053695` | `0.043924` |
| Worked mathematics, science, and formal reasoning | `0.102133` | `0.086180` |

Dialogue, instruction, and assistance is the largest-divergence stratum in both
partitions. Its source-cluster 95% interval is `0.765049`–`0.989871` for 96
development-facing contexts and `0.615515`–`0.924076` for 32 held-out contexts.
This subgroup result is exploratory because the evaluation contract names the
aggregate natural-route KLD as the primary endpoint and assigns no subgroup
acceptance threshold.

## Interpretation

The two shared-route checkpoint comparisons agree closely despite using route
sequences captured from different checkpoints. This supports a conditional
checkpoint effect near `0.067`–`0.071` nats/token for the evaluated suite and
runtime.

Natural execution has materially greater distribution divergence than either
shared-route execution. Natural BF16 and NVFP4 routes agree exactly at the first
router and diverge progressively after the first quantized expert computation.
Together, these observations support a route-feedback mechanism: an early
numeric perturbation changes later hidden states, which changes later router
choices, which then further changes hidden states. The design does not identify
an additive percentage of total KLD attributable to routing because KLD is
directional and the checkpoint and route interventions interact.

The dialogue-stratum concentration warrants a separate diagnostic before any
deployment conclusion. Appropriate follow-up measurements include
prompt-template position analysis, per-source review, downstream task
evaluation, and generated-sequence comparisons. Aggregate KLD cannot determine
whether high-divergence positions correspond to better, worse, or behaviorally
equivalent outputs.

## Validation

- Repeated controlled BF16 captures of the same context are bit-identical.
- Repeated NVFP4 `FLASHINFER_CUTLASS` captures of the same context are
  bit-identical in hidden states and all logical route payloads.
- Replaying a captured BF16 hidden state through the shared BF16 language-model
  head reproduces live BF16 logits exactly for all 2,047 scored positions in
  `context-0000`.
- A synthetic 10,003-class comparator test agrees with a direct FP64 oracle
  within `3.2e-8` for KLD and `2.5e-8` for Jensen-Shannon divergence; identical
  distributions produce exact zero.
- Position blocks of 2,048 and 128 produce identical metrics for the same real
  context.
- Capture and result validation verifies token IDs, tensor shapes, suite hashes,
  language-model-head hash, and source artifact hashes.

## Limitations

- The result applies to one BF16 revision, one NVFP4 revision, one runtime
  image, and the `FLASHINFER_CUTLASS` MoE backend on NVIDIA Streaming
  Multiprocessor 120 (SM120) GPUs.
- B12X NVFP4 inference must become reproducible and be measured separately
  before these values can be attributed to a B12X deployment.
- The shared BF16 language-model head intentionally excludes output-head
  quantization from the comparison.
- The suite measures teacher-forced next-token distributions over fixed prompt
  prefixes. It does not measure autoregressive trajectory divergence after the
  checkpoints sample different tokens.
- KLD has no universal conversion to task accuracy, preference, safety, or
  perceived response quality.
- The source-cluster bootstrap represents uncertainty over the sampled source
  clusters. It does not establish population coverage for all deployment
  traffic.
- Route-controlled executions are counterfactual interventions and are not
  normal serving behavior.
- Full captures and per-position arrays are retained by Local Inference Lab but
  are not published in this Git repository. The receipt hashes below identify
  the retained result objects.

## Receipt identities

| Receipt | `analysis` SHA-256 | `qualification` SHA-256 |
|---|---|---|
| Natural BF16 to natural NVFP4 | `2f094ff0eb5413c4a5460a3aa6ea4fe74e23a9323083962bcc80fe0ce3aadd4c` | `9189238d63429ca27812e05f86521b50b02986ae7855a223ec45a14f62159fb3` |
| Both using BF16 routes | `a9d23e168d9b2839dc92d1b32b3f3af4646be2f45ecec048b20935c46d2075c8` | `7e59f2496b06f91c64d5cd71d472ca54efc5e9e3b71a9a9b8f5440ac2f561429` |
| Both using NVFP4 routes | `e784e711d524e4d835234d03e91cc4212634d9004d5f2fc7632ec14b260275fc` | `96cb9871cf2d1bf88454d5f80d9ad661f8f3ead301181e73e47f6f87a5c34063` |
| BF16 route sensitivity | `2c4ec7714dbdcc6f06b112ddb7b0b19c41655563dda2ce548ca456811ede064f` | `78fe881d56403a9e6fa101c94be8c98bfca836c225ce99d1b33ca647b42512b6` |
| NVFP4 route sensitivity | `52c4838570044689c7738abf6839c4a659ad418b38d3326be80ea550086c4f27` | `fd71fbeb0cd85d5344f1d9df3a3d495ad9759c9d08109b5d593971e7d8f3d461` |
| Natural route agreement | `65f201ae1cefcf760721e8b6a3c1b676fa6c2c765bfe1d033eaec0f8cc53b405` | `78b6e2857509ac983b49dc0653f81b7ea46fcc954961df8704d2346d76a32a47` |

## Contribution record

The [general KLD protocol](README.md#scientific-and-technical-provenance)
records the scientific precedents and Local Inference Lab contribution history.
For this GLM-5.3-Flash measurement, Luke Alonso defined the requirement to
separate fixed-route checkpoint divergence from natural-route behavior. Martin
Vit (`Festr`) constructed the GLM-specific suite, implemented and validated the
model-specific route capture and replay, executed the four inference cells,
computed the distribution and route statistics, and assembled the receipts and
report. AI tools assisted implementation and documentation under human
direction and review.
