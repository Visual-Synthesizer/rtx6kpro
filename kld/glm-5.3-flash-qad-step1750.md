# GLM-5.3-Flash NVFP4 QAD step 1,750 distribution fidelity

Status: **research-only**.

This report evaluates the Local Inference Lab
`GLM-5.3-Flash-NVFP4-QAD-step1750` checkpoint against the pinned Brain
Floating Point 16-bit (BF16) reference and the published NVIDIA 4-bit
floating-point (NVFP4) comparator. The checkpoint is a
quantization-aware-distillation (QAD) materialization produced by the Quatrain
training workflow. It retains the NVFP4 storage format used by the comparator
while replacing trained routed-expert weights, tensor scales, shared-expert
weights, and normalization weights.

The pre-specified primary endpoint is forward Kullback-Leibler Divergence
(KLD) under exact BF16-route replay. On 524,020 held-out positions, QAD reduces
token-weighted KLD from `0.067476` to `0.060727` nats/token, a `10.001%`
reduction. The paired, allocation-stratified source-cluster 95% bootstrap
interval for the change is `-0.008970` to `-0.004945`; its upper endpoint is
below zero, so the primary directional criterion passes.

The pre-specified secondary natural-route endpoint also passes. KLD decreases
from `0.162164` to `0.144644` nats/token, a `10.804%` reduction, with paired
95% interval `-0.022871` to `-0.012784`.

The separate
[AA-LCR capability comparison](../models/glm-5.3-flash/aa-lcr-nvfp4-vs-qad-step1750.md)
scores the published NVFP4 checkpoint at 74.00% and QAD step 1,750 at 73.00%.
Its paired interval includes zero. The distribution-fidelity improvement
therefore does not establish a measurable AA-LCR improvement on the evaluated
100-question sample.

These aggregate results do not define a deployment-quality threshold. The
improvement is heterogeneous and concentrated in dialogue/instruction data.
QAD has lower per-position KLD on fewer than half of scored positions, so the
mean reduction is driven by the magnitude and tail of the errors rather than a
uniform token-level improvement.

The separate
[Verifier-Backed Behavioral Fidelity comparison](../models/glm-5.3-flash/verifier-backed-behavioral-fidelity.md)
scores BF16 at 93.11%, published NVFP4 at 91.63%, and QAD step 1,750 at 92.35%
on deterministic tasks with executable answer keys. QAD has the closer
fractional point estimate, but the paired interval does not establish
behavioral improvement, non-inferiority, or equivalence within one percentage
point.

## Checkpoint identity and contents

| Role | Artifact identity |
|---|---|
| BF16 distribution reference | `zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a` |
| Published NVFP4 comparator | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |
| QAD candidate | `GLM-5.3-Flash-NVFP4-QAD-step1750`; Quatrain checkpoint step 1,750; checkpoint-manifest SHA-256 `ed76ada3ee9e4bf10d74554fb62a8d0e1767e8d5b22ee15f4940cc10cbf2da5c` |

The QAD materialization manifest has SHA-256
`4c8439a5e8891f0dc7f22134803e44aebd4296284e356ae0a8bb2cf76e4ee937`.
Its weight index has SHA-256
`b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b`.
All 44 indexed files were checked against the sizes and SHA-256 values in the
materialization manifest before inference. They contain 148,498 indexed
tensors and 198,042,331,512 indexed tensor bytes.

The materialization contract separates trained and inherited state:

- Decoder layers 3 through 44 contain trained routed-expert projections
  serialized as NVFP4 E2M1 values with finite E4M3 block-16 scales and trained
  32-bit floating-point (FP32) tensor scales.
- Trained shared-expert and Root Mean Square Normalization (RMSNorm) weights
  retain BF16 storage. The manifest records 217 trained replacements outside
  the routed-expert shard set.
- Attention, routers, dense multilayer perceptrons, embeddings, language-model
  head, vision components, tokenizer, chat template, and generation assets are
  inherited from the published NVFP4 comparator.
- The Multi-Token Prediction (MTP) routed experts retain the comparator's
  microscaling 8-bit floating-point (MXFP8) `[1, 32]` storage. MTP is disabled
  in this teacher-forced target-model evaluation.
- The routed-expert input-scale file is bit-identical to the published NVFP4
  comparator. Its SHA-256 is
  `4255779f031450572af8548c610fd9abfe7df89704985d18595c21788593cd05`.

The result therefore measures a trained checkpoint, not a recalibration-only
variant and not a change to tokenizer or prompt formatting.

The resident Quatrain training objective combines full-vocabulary forward KLD
from the BF16 teacher with mean-squared error on the final hidden state that
enters the shared language-model head. The serialized artifact is the NVFP4
materialization of training step 1,750, not the resident training
representation itself.

## Evaluation design

### Token suite

The evaluation suite has durable identifier
`glm-5.3-flash-kimi-k3-source-fidelity-1024x-max2048-v1`. It contains 1,024
GLM-tokenized contexts with at most 2,048 prompt tokens across ten allocation
strata. Dialogue sources use the pinned GLM chat template. Source clusters do
not cross the partition boundary.

| Partition | Contexts | Scored next-token positions | Role |
|---|---:|---:|---|
| `analysis` | 768 | 1,571,435 | Development-facing measurement and hypothesis formation |
| `qualification` | 256 | 524,020 | Held-out evaluation read after the decision rules were frozen |

The suite manifest SHA-256 is
`50520bdba81a9447b769e72da58720fb8468bb6dd19f641493c9110b04f9972b`.
The ordered token hash is
`e2c541bce4a3213f697cd5236eaa60393d044dc054ea3ec2cc35d79dcb089f9b`.
The source selection comes from the public
[`festr2/kimi-k3-distribution-fidelity-1024x2048-v1`](https://huggingface.co/datasets/festr2/kimi-k3-distribution-fidelity-1024x2048-v1)
artifact at revision `402919ae70d61396087571b63fe9185d95491afb`, but
the token artifacts are GLM-specific. Kimi-K3 hidden states are not used as a
GLM reference.

### Distribution and route conditions

Every captured final hidden state is projected through the same BF16
language-model head, whose safetensors SHA-256 is
`ace852f317fb56d240f539e4dbffc0883827cfc1f58b3df5ba2d389289801869`.
The comparator reconstructs all 154,880 logits at every scored position and
computes full-vocabulary forward `KL(BF16 || candidate)` without top-k or
top-p truncation.

Two execution conditions answer different questions:

| Condition | BF16 operand | NVFP4 operand | Estimand |
|---|---|---|---|
| Natural routes | Uses its own selected expert IDs and weights | Uses its own selected expert IDs and weights | End-to-end distribution fidelity of the executed checkpoint |
| Exact BF16 routes | Uses its natural selected expert IDs and weights | Replays the same logical expert IDs and consumed route weights | Checkpoint fidelity conditional on one shared expert path |

The exact route trace covers the selected top-eight experts and weights for
every prompt token and every Mixture-of-Experts (MoE) layer from layer 3
through layer 44. Replay occurs after top-k selection and before Expert
Parallel Load Balancing (EPLB) remapping. Fixed-route and natural-route values
are counterfactual and deployed estimands respectively; they are not additive
components of KLD.

### Runtime and topology

The BF16 reference capture uses Tensor Parallelism 8 (TP8). Both NVFP4
checkpoints are executed with TP4 on the same four NVIDIA RTX PRO 6000
Blackwell GPUs. Pairing therefore compares the two candidates against exactly
the same BF16 distribution while matching candidate topology and runtime.
Absolute BF16-to-candidate KLD includes the declared BF16-versus-candidate
topology difference; the paired QAD-minus-comparator contrast does not compare
TP4 against TP8 candidates.

The separately published
[BF16-to-NVFP4 report](glm-5.3-flash-bf16-nvfp4.md) uses TP8 for its NVFP4
operand. The topology-matched TP4 comparator in this report changes
natural-route KLD from `0.176065` to `0.175257` on `analysis` and from
`0.161911` to `0.162164` on `qualification`. Exact-BF16-route KLD changes from
`0.070211` to `0.070514` and from `0.066807` to `0.067476` respectively. The
aggregate differences are small, but the captures are not bit-identical; every
QAD ranking on this page uses the TP4 comparator.

Candidate inference uses eager execution, disabled Torch compilation,
sequential shared-expert execution, NVIDIA Collective Communications Library
(NCCL) tensor-parallel reductions, disabled custom all-reduce, and the
reproducible `FLASHINFER_CUTLASS` MoE backend. The instrumented image is
`local/glm53-kld-r18:dev-20260903`, with image ID
`sha256:11a0ad1530f8232050e7bff18350860421e5815e645026905135cde3e1cfff73`.

The B12X kernel/backend stack's NVFP4 MoE backend is **unsupported for this
measurement**. Its real layer-3 kernel is nondeterministic under the available
deterministic controls, so no KLD value in this report characterizes the B12X
serving path.

### Frozen decision rules

The qualification contract was written before either TP4 checkpoint executed
the 256 qualification contexts. Its SHA-256 is
`262ecc1cbc3566f5c63a3636fd6782785b8378a5d0a63594f4ac05804e7968fc`.

The primary endpoint is the paired QAD-minus-published-NVFP4 change in
token-weighted KLD under exact BF16-route replay. The secondary endpoint is the
same paired change under natural routes. Each directional claim passes only
when the upper endpoint of a 10,000-sample paired bootstrap interval is below
zero. The bootstrap resamples `(dataset, source_cluster_id)` units within each
allocation stratum, preserves observed stratum position weights, and uses seed
1. No absolute KLD pass threshold is defined.

## Paired fidelity results

Negative changes favor QAD. The interval is the pre-specified paired,
allocation-stratified source-cluster micro interval.

| Partition | Routing condition | Published NVFP4 KLD | QAD KLD | QAD minus comparator | Relative reduction | Paired 95% interval | Criterion |
|---|---|---:|---:|---:|---:|---:|---|
| `analysis` | Natural | `0.175257` | `0.156552` | `-0.018705` | 10.673% | `-0.021784` to `-0.015804` | Development-facing evidence |
| `analysis` | Exact BF16 routes | `0.070514` | `0.062700` | `-0.007814` | 11.081% | `-0.008844` to `-0.006725` | Development-facing evidence |
| `qualification` | Natural | `0.162164` | `0.144644` | `-0.017521` | 10.804% | `-0.022871` to `-0.012784` | Pass |
| `qualification` | Exact BF16 routes | `0.067476` | `0.060727` | `-0.006748` | 10.001% | `-0.008970` to `-0.004945` | Pass |

The held-out partition confirms both directional hypotheses. The analysis and
qualification effect sizes have the same sign and similar scale.

### Held-out secondary metrics and tails

| Routing condition | Checkpoint | Mean JS | Top-1 agreement with BF16 | KLD p99 | KLD p99.9 | Maximum KLD |
|---|---|---:|---:|---:|---:|---:|
| Natural | Published NVFP4 | `0.023748` | 90.663% | `3.009845` | `14.564972` | `32.544923` |
| Natural | QAD | `0.022990` | 90.788% | `2.595301` | `12.483213` | `30.669209` |
| Exact BF16 routes | Published NVFP4 | `0.013388` | 93.034% | `1.118111` | `4.900694` | `33.261937` |
| Exact BF16 routes | QAD | `0.012904` | 93.217% | `1.010593` | `3.650674` | `27.426561` |

QAD improves mean Jensen-Shannon divergence, vocabulary top-1 agreement, and
the reported high-KLD tails in both routing conditions. These are descriptive
secondary metrics, not additional pass criteria.

## Allocation-stratum heterogeneity

The table reports held-out QAD-minus-comparator KLD changes and independent
within-stratum source-cluster 95% intervals. Negative values favor QAD.

| Allocation stratum | Natural-route change (95% interval) | Exact-BF16-route change (95% interval) |
|---|---:|---:|
| Chinese | `-0.005391` (`-0.014998` to `0.004466`) | `-0.008273` (`-0.015559` to `-0.002750`) |
| Code, tests, documentation, and issues | `-0.002275` (`-0.004574` to `-0.000432`) | `-0.000658` (`-0.001747` to `0.000379`) |
| Dialogue, instruction, and assistance | `-0.119980` (`-0.162890` to `-0.084937`) | `-0.039939` (`-0.056984` to `-0.027030`) |
| Encyclopedic and factual | `-0.007969` (`-0.012194` to `-0.004231`) | `-0.003292` (`-0.005868` to `-0.000829`) |
| Literary, narrative, and creative | `0.000541` (`-0.001889` to `0.003427`) | `-0.000680` (`-0.001748` to `0.000743`) |
| News, history, economics, legal, and essays | `0.000335` (`-0.002444` to `0.004833`) | `-0.001320` (`-0.001936` to `-0.000928`) |
| Other multilingual | `-0.002806` (`-0.006162` to `-0.000102`) | `-0.001754` (`-0.005425` to `0.000761`) |
| Scientific and technical | `-0.000057` (`-0.000936` to `0.000883`) | `-0.000613` (`-0.001571` to `0.000082`) |
| Structured data, tools, APIs, and tables | `-0.002736` (`-0.004647` to `-0.001065`) | `-0.001473` (`-0.002495` to `-0.000229`) |
| Worked mathematics, science, and formal reasoning | `-0.004416` (`-0.007119` to `-0.000548`) | `-0.000570` (`-0.002348` to `0.001193`) |

Dialogue, instruction, and assistance contains 12.5% of held-out positions but
accounts arithmetically for 85.6% of the net natural-route reduction and 74.0%
of the net fixed-route reduction. Several other strata improve, while several
intervals include zero. The aggregate reduction must not be generalized as a
uniform 10% improvement across domains.

At the individual-position level, QAD has lower KLD than the comparator on
49.69% of natural-route positions and 49.28% of exact-BF16-route positions.
The median paired change is approximately zero in both conditions. Large error
reductions on a minority of positions outweigh small regressions elsewhere.

## Natural-route behavior

Route metrics cover every scored token and all 42 routed MoE layers. Route
weight Total Variation (TV) is computed after independently normalizing each
top-eight weight vector.

| Partition | Pair | Top-1 expert agreement | Ordered slot agreement | Mean top-8 set overlap | Exact top-8 set agreement | Mean weight TV |
|---|---|---:|---:|---:|---:|---:|
| `analysis` | BF16 vs published NVFP4 | 89.375% | 55.442% | 87.251% | 36.613% | `0.112382` |
| `analysis` | BF16 vs QAD | 89.470% | 55.583% | 87.348% | 36.771% | `0.111505` |
| `analysis` | Published NVFP4 vs QAD | 89.798% | 57.094% | 87.807% | 39.065% | `0.107635` |
| `qualification` | BF16 vs published NVFP4 | 89.308% | 55.462% | 87.172% | 36.641% | `0.113143` |
| `qualification` | BF16 vs QAD | 89.409% | 55.623% | 87.270% | 36.822% | `0.112229` |
| `qualification` | Published NVFP4 vs QAD | 89.762% | 57.176% | 87.758% | 39.174% | `0.108116` |

QAD route agreement with BF16 is modestly better than the comparator on both
partitions, but most route decisions after the first quantized MoE layer are
not identical. This supports reporting both natural and fixed-route results;
it does not provide an additive decomposition of distribution error.

## Direct checkpoint-to-checkpoint distance

Forward KLD is asymmetric. The following descriptive comparison uses the
published NVFP4 distribution as the reference, so it is
`KL(published NVFP4 || QAD)` and is not a quality score.

| Partition | Routing condition | Direct KLD | Mean JS | Vocabulary top-1 agreement |
|---|---|---:|---:|---:|
| `analysis` | Natural | `0.152893` | `0.023045` | 90.633% |
| `analysis` | Exact BF16 routes | `0.063740` | `0.012651` | 93.112% |
| `qualification` | Natural | `0.142029` | `0.022013` | 90.769% |
| `qualification` | Exact BF16 routes | `0.061797` | `0.012421` | 93.148% |

The direct values show that QAD materially changes the next-token distribution
of the NVFP4 comparator. They do not establish whether either changed token
probability is behaviorally preferable.

## Training-evaluation separation

The retained QAD replay store contains 41,983 native GLM rollout documents and
124,940,998 tokens. Its source tags comprise 12,191 diverse rollouts and
29,792 coding-oriented rollouts. The replay-store manifest SHA-256 is
`bd7370994a52979c518ced5b01f3be5edef269ca11dbb31eaa7b1fc569873efb`.

An exact-token audit scanned all replay tokens against all 1,024 evaluation
contexts. It found no shared 256-token context prefix and no exact full-context
match. The overlap receipt SHA-256 is
`3efdec7aca63f4dc9fc994e1b0252639e91961cd978dfd730eeb9a5093155cad`.
The audit does not detect semantic similarity, paraphrases, or shared spans
shorter than 256 tokens, so it is not evidence of complete contamination
absence.

The checkpoint card's approximate `KLD: ~0.04` is not a result from this
evaluation and must not be compared numerically with the values above. The
training workflow's fixed 130,086-token probe at step 1,750 reports `0.020459`
under its teacher-route configuration and `0.059343` under student routes. The
teacher-route probe combines teacher expert IDs with student sigmoid route
weights rather than replaying the exact teacher IDs and consumed weights. Its
dataset, route intervention, resident training representation, and aggregation
contract differ from this report.

## Validation

- Independent QAD server starts produced identical canonical tensor payloads
  for one 2,048-token context. The final-hidden-state payload SHA-256 is
  `54bd24baf9919148df8bf4f6c2ce0296ab822e121e52c8f0f7bf6368d014708e`,
  and the 42-layer logical-route payload SHA-256 is
  `fbc15f11d0ffec2011c7b8b72c98c867a2abd0882d0e219313451fa7e3dc281a`.
- The candidate captures contain exactly 768 analysis contexts and 256
  qualification contexts in each routing condition. Every hidden-state and
  route source hash was verified during scoring.
- The comparator uses a full-vocabulary row-maximum pass, float32 shifted
  exponentials, float64 mass accumulation, and float64 weighted reductions.
  Its SHA-256 is
  `54b1ab1c24ca5d671c00fcd27f693f8a489024a55521b34f8bfeb12d27f20b7b`.
- The paired comparator SHA-256 is
  `8a1056da99c216861459caf92f2216d0db1b4d225981b67bc3fcafc887863087`.
  The route comparator SHA-256 is
  `85ad17d50f0f0b81f68a1f97061c70bc2bf8236085097857621ffcfff0282879`.
- No scored comparison produced a negative KLD or Jensen-Shannon value before
  the configured numerical roundoff clamp.

## Interpretation and limitations

The evidence supports a narrow conclusion: for the pinned artifacts, suite,
shared BF16 head, and reproducible `FLASHINFER_CUTLASS` TP4 candidate runtime,
QAD step 1,750 has lower aggregate forward KLD to the BF16 reference than the
published NVFP4 comparator under both exact BF16-route replay and natural
routing. The conclusion replicates on the held-out partition under decision
rules fixed before qualification inference.

The evidence does not establish a universal 10% fidelity improvement, task
accuracy, preference quality, safety, serving throughput, or B12X behavior.
Additional limitations are:

- Teacher-forced next-token KLD does not measure divergence after two models
  sample different autoregressive continuations.
- A common BF16 language-model head intentionally removes independently
  rounded output-head effects from the comparison.
- The QAD replay corpus is rollout-heavy and coding-heavy, while the largest
  measured gain occurs in the dialogue/instruction evaluation stratum.
  Workload-specific capability and generation tests remain necessary.
- The exact-token overlap scan does not cover semantic or short-span overlap.
- The BF16 operand uses TP8 while both candidates use TP4. This is controlled
  for the paired candidate ranking but limits interpretation of absolute KLD
  as a topology-independent checkpoint property.
- The suite represents a declared ten-stratum sample, not all possible GLM
  deployment traffic.
- Full captures and per-position arrays are retained by Local Inference Lab but
  are not published in the documentation repository.

## Receipt identities

| Receipt | `analysis` SHA-256 | `qualification` SHA-256 |
|---|---|---|
| BF16 to published NVFP4, natural routes | `ae25182bdd4783fbf1e3540f021202199471274ec59a53c55d4ca3a694dfdeca` | `d9eafbd9eb712dc4fb19aa97fba7f9458d604ff293a4f107638f686fbe3430f5` |
| BF16 to QAD, natural routes | `a38cc114731d440d5e808ffaf6b79e5e5b2afdfdfa29596fb774bf53b4172469` | `4ce88d7683a4874c713a138fc432f755b369b409bdeaeae848632e30f5ca1fdf` |
| Paired QAD-minus-comparator, natural routes | `33d0eeecc979e9bd6dac7f33f2224b734bf46ad8b92d0fe25584cfabc210975f` | `526e02b2ae4251ea2d9452aa25c4ed04464f0d3b5203ba0442298889458f4019` |
| BF16 to published NVFP4, exact BF16 routes | `d1d59fe2b53f84207c364014b2e624c00706cdaeefb1f0d647127ce8d087e499` | `2a52d0c286d2d7f6aed09ce47021e4ed4b7637de80725690ffeb601beb3e6a73` |
| BF16 to QAD, exact BF16 routes | `e7115bc5cce8c7689cf728377163795ec9c4abf60db69f89b59e6ea8727ebf61` | `f688cf7aab8050301f415c9a4ba105fd545a65b7cb4a3a7cd2a47e6924b6c217` |
| Paired QAD-minus-comparator, exact BF16 routes | `18f3fbc58763ab591f83526e1bb4f23d6bcbb571dd03b5f060c2cc9f4d4016e3` | `923f9750ec30c1c5e8e031c549a42248dd965b0e886aef41d8499ede47cee56e` |
| Published NVFP4 to QAD, natural routes | `ab5e74c94bcbf925ecc63deecc71a689c36e1995771153e84421505edefdb7d6` | `7266145ad4dc0e61c21f7736e919997a7775be9b06c05e6ef86cf3cde235d08a` |
| Published NVFP4 to QAD, exact BF16 routes | `499bd57a2ad53053a990d5f37f746609b775c9ce8c210a32b573b17f68536d66` | `5e2e482d57550dd2757c0e3537f9a52f6eb2a24a8bdb0ad27a8dcb8fb4490957` |
| Natural routes, BF16 versus published NVFP4 | `04137ff01d0b4df9c1705349a502e099631e92498d3f7779b423ec14913446c2` | `2db0f362dd87ae9a87b914ca29e843707b7d7583f855785165675cb27fac2c55` |
| Natural routes, BF16 versus QAD | `c32c28779be0ab0a169117b12bcefd14989d26a7655f2d22c12616ab857901c2` | `e70bb4d83b0031cb094a0ed2a4d4963781452b0dd0d61a59f9f8780bbf2812c8` |
| Natural routes, published NVFP4 versus QAD | `f6ba04900d4b79931956ecce1c5996543c9e1949ef57b96b861611dcb9b296c2` | `b21dd7aa72c46d1177bbe59a7019617477f63b7b47e70bcdebdf3009f4668ca3` |

## Contribution record

The Local Inference Lab [general KLD protocol](README.md#scientific-and-technical-provenance)
records the scientific precedents and contribution history. Phaelon's
full-vocabulary vLLM inference-engine score-mode work provided the lab's
initial engineering inspiration. Luke Alonso directed the shared-head fidelity
program and defined the requirement to rank MoE quantization checkpoints under
fixed baseline routes as well as natural routes. Martin Vit (`Festr`)
constructed and validated the GLM token suite, implemented and qualified the
capture, exact route replay, distribution comparison, pairing, overlap audit,
and receipt
tooling, executed the measurements, and assembled this report.

The evaluated checkpoint is credited to the Local Inference Lab Quatrain
training and materialization workflow; the retained artifact manifest does not
assign individual authorship. AI tools assisted implementation and
documentation under human direction and review. These credits do not claim
invention of KLD, quantization-aware distillation, hidden-state caching, or
route-controlled causal designs.
