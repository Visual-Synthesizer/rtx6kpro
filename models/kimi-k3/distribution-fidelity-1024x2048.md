# Kimi K3 distribution-fidelity reference: 1,024 contexts × 2,048 tokens

## Status

| Component | Status | Evidence |
|---|---|---|
| Token-ID evaluation suite | implemented | 1,024 structurally validated contexts; suite token hash `70cd72175fcb4574123b90a83e2b71bb460b8c2938cec9573645dc8c9a46a3bd` |
| Hidden-state LM-head replay | qualified | The 32-context live-logit suite was reproduced with mean replay KLD `1.229325e-6` and top-1 agreement `0.999954` |
| Official MXFP4 reference hidden states | implemented | 1,024 BF16 tensors with shape `[2047, 7168]`; file hashes are recorded in `reference-hidden/manifest.json` |
| Runtime-repeat sentinels | implemented | 64 stratified contexts captured three times as hidden states and live BF16 logits |
| Candidate comparison | implemented | Full-vocabulary two-pass KLD, Jensen–Shannon divergence, top-1 agreement, stratified context bootstrap, depth buckets, and paired candidate reports |
| Capability and long-context evaluation | unsupported | The artifact measures teacher-forced distribution fidelity at a 2,048-token context only |

The downloadable artifact is
[`festr2/kimi-k3-distribution-fidelity-1024x2048-v1`](https://huggingface.co/datasets/festr2/kimi-k3-distribution-fidelity-1024x2048-v1).

## Measurement

For aligned token IDs, the primary metric is

```text
KL(official MXFP4 distribution || candidate distribution)
```

The runtime captures the BF16 transformer output after final RMSNorm and
immediately before the language-model head. One shared BF16 language-model head
then reconstructs complete 163,840-token distributions for both operands. This
removes approximately 612 GiB of reference storage relative to retaining BF16
full-vocabulary logits for every position.

The artifact does not measure free-running generation, benchmark correctness,
multimodal behavior, tool execution, or behavior beyond 2,048 input tokens.
Those properties require separate evaluations.

## Immutable identity

| Item | Durable identity |
|---|---|
| Dataset layout | `kimi-k3-distribution-fidelity-1024x2048-v1` |
| Contexts | 1,024 distinct source documents |
| Source clusters | 827 dataset-qualified document families; software contexts use one repository each |
| Tokens per context | 2,048 |
| Scored positions per context | 2,047 |
| Total scored positions | 2,096,128 |
| Hidden width | 7,168 |
| Vocabulary size | 163,840 |
| Suite manifest SHA-256 | `f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf` |
| Suite token SHA-256 | `70cd72175fcb4574123b90a83e2b71bb460b8c2938cec9573645dc8c9a46a3bd` |
| Official checkpoint | `moonshotai/Kimi-K3` revision `2496450e92e425c886db095102a52a6682ca3970` |
| Official checkpoint index SHA-256 | `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| LM-head safetensors SHA-256 | `c282a205aed4d19fbbe4ff907feff979c50b24723e8f552ddc141ef008c5de11` |
| LM-head raw tensor SHA-256 | `11c1f1c09a8e0db55547b5e68ebfd1d8e3b503bee56c4e1312ef55ecd3e5580f` |
| Capture vLLM commit | `e77ee0612b9b7d117439920ef81bdbb162d09cd3` |
| Runtime image digest | `voipmonitor/vllm@sha256:974edc237f27a4eaa83a53ce4927dd176a5ad8ce4fbb8d3d689fce82348531a5` |
| Runtime manifest SHA-256 | `ec02e1633b045d9e7d525b41a02f20b2e1ab08adceabb4338a678b716039a783` |

`checksums.txt` is authoritative for downloadable file integrity. Candidate
captures must use the stored token IDs directly; retokenizing source text does
not reproduce the evaluation input.

## Content allocation

| Allocation stratum | Contexts |
|---|---:|
| Encyclopedic and factual reference | 128 |
| Scientific and technical exposition | 128 |
| News, history, economics, legal analysis, and essays | 96 |
| Literary, narrative, and creative writing | 96 |
| Natural dialogue, instruction following, and assistance | 128 |
| Source code, tests, technical documentation, and issue discussions | 128 |
| Worked mathematics, science, and formal reasoning | 128 |
| Chinese across several content types | 96 |
| Other multilingual content | 48 |
| Structured data, tool calls, APIs, JSON, and tables | 48 |

Every context comes from a distinct coherent source unit. Source identity,
dataset revision, extraction policy, content hash, deterministic token offset,
representation type, tokenizer identity, and final token hash are recorded in
`sources.json`, `source-registry.json`, and `suite-manifest.json`. Exact token
deduplication and approximate token-shingle MinHash deduplication run before
partition assignment.

`validation/capability-overlap.json` records a normalized exact-content scan
against HumanEval revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`,
MMLU-test revision `c30699e8356da336a370243923dbaf21066bb9fe`, and
GPQA-Diamond mirror revision `acc659161e28d4416c0ed44e3fc85c9383add471`.
The scan found zero complete benchmark-item overlaps. For MMLU, a match requires
the question and all answer choices, which avoids treating a short general
question appearing naturally in reference prose as benchmark leakage.

The deterministic partitions contain 768 analysis contexts and 256
qualification contexts. A stratified subset of 64 analysis contexts is marked
as runtime-repeat sentinels. Quantization parameters and acceptance thresholds
must be frozen from analysis-partition results before qualification-partition
results are used.

## Artifact layout

```text
manifest.json
suite-manifest.json
capture-runtime.json
source-registry.json
sources.json
partitions.json
tokens/context-0000.json ... context-1023.json
reference-hidden/hidden_0000.safetensors ... hidden_1023.safetensors
lm-head/weight.safetensors
sentinel-hidden/repeat-01/
sentinel-hidden/repeat-02/
sentinel-live-logits/repeat-00/
sentinel-live-logits/repeat-01/
sentinel-live-logits/repeat-02/
capture-tools/
comparators/
validation/
checksums.txt
```

Each reference hidden-state file contains one BF16 tensor named
`hidden_states` with shape `[2047, 7168]`. The LM-head file contains one BF16
tensor named `weight` with shape `[163840, 7168]`.

## Download

```bash
hf download festr2/kimi-k3-distribution-fidelity-1024x2048-v1 \
  --repo-type dataset \
  --local-dir /mnt/kld/kimi-k3-distribution-fidelity-1024x2048-v1
```

Use `sha256sum --check checksums.txt` from the artifact root before capture or
comparison. A partial download may omit the 120 GiB `sentinel-live-logits/`
directory when only candidate KLD is required. Runtime-repeat analysis needs
the sentinel files.

## Capture implementation

The vLLM capture implementation is the two-commit range
`881ac39a4fb6c5bbfa14f3944db560e0a27f3ffe..e77ee0612b9b7d117439920ef81bdbb162d09cd3`.
The exact source is published on
[`research/kimi-k3-kld-hidden-capture-20260813`](https://github.com/local-inference-lab/vllm/tree/research/kimi-k3-kld-hidden-capture-20260813).
It adds an environment-gated V2-runner hook that:

- captures the tensor after final RMSNorm and before the language-model head;
- writes only from global rank zero after tensor-parallel assembly;
- writes bounded BF16 safetensors chunks during prefill;
- does not change normal serving when `VLLM_KLD_HIDDEN_CAPTURE_DIR` is unset;
- refuses ambiguous or overlapping request captures.

The source files and their hashes are recorded in `capture-runtime.json`. The
capture server verifies the exact source commit and Docker image ID before
launching:

```bash
export CAPTURE_VLLM_SOURCE=/path/to/vllm-at-e77ee0612b9b7d117439920ef81bdbb162d09cd3
export CAPTURE_OUTPUT=/mnt/kld/candidate-capture-runtime
export CAPTURE_HF_CACHE=/root/.cache/huggingface
export CAPTURE_JIT_CACHE=/path/to/writable-kimi-k3-jit-cache

models/kimi-k3/tools/serve-kimi-k3-fidelity-capture.sh
```

The default checkpoint is the official MXFP4 snapshot. A candidate stored in
the same Hugging Face cache can be selected by setting
`CAPTURE_MODEL_RELATIVE` to its path relative to `CAPTURE_HF_CACHE`.

The qualified server configuration is:

| Setting | Value |
|---|---|
| Tensor parallelism / decode context parallelism | TP16 / DCP1 |
| Expert weights and compute | official MXFP4 W4A16 |
| Activation and KV-cache dtype | BF16 |
| Attention | B12X MLA |
| KDA prefill | Triton |
| MoE and dense linear backend | B12X |
| Weight loader | InstantTensor |
| Maximum model length | 4,096 |
| Maximum batched tokens | 256 |
| Maximum sequences | 1 |
| KV allocation | 300,000,000 bytes per rank |
| CUDA graph mode | piecewise, capture size 1 |

Runtime logs must prove the checkpoint identity, quantization path, B12X
backends, KDA backend, TP/DCP topology, activation dtype, and loader before a
capture receipt is accepted.

## Capture a candidate

Create a candidate runtime manifest that records the candidate checkpoint,
container, source revisions, topology, backend selections, dtypes, and relevant
environment variables. Keep every serving parameter equal to
`capture-runtime.json` except the checkpoint and quantization-specific loader
configuration.

After the server health endpoint returns success, run:

```bash
python models/kimi-k3/tools/capture-kimi-k3-hidden-suite.py \
  --url http://127.0.0.1:8001/v1/completions \
  --model Kimi-K3 \
  --suite-dir /mnt/kld/kimi-k3-distribution-fidelity-1024x2048-v1 \
  --capture-dir /mnt/kld/candidate-capture-runtime/capture-hidden \
  --output-dir /mnt/kld/candidate-hidden \
  --runtime-manifest /mnt/kld/candidate-runtime.json \
  --run-name candidate-checkpoint-identifier \
  --context-filter analysis \
  --delete-raw-chunks-after-finalize
```

The client is resumable. It validates and skips every recorded context whose
token hash, tensor metadata, file hash, shape, and dtype match its manifest.
Use `--context-filter qualification` only after the candidate configuration and
acceptance rule are frozen.

## Compute full-vocabulary metrics

The comparator performs two vocabulary passes in bounded chunks. It computes
full log-sum-exp normalizers first, then exact reference-to-candidate KLD and
Jensen–Shannon divergence. It never substitutes top-k logits for the complete
vocabulary.

```bash
python models/kimi-k3/tools/compare-kimi-k3-hidden-replay.py \
  --reference-hidden-dir /mnt/kld/kimi-k3-distribution-fidelity-1024x2048-v1/reference-hidden \
  --candidate-hidden-dir /mnt/kld/candidate-hidden \
  --lm-head /mnt/kld/kimi-k3-distribution-fidelity-1024x2048-v1/lm-head/weight.safetensors \
  --suite-manifest /mnt/kld/kimi-k3-distribution-fidelity-1024x2048-v1/suite-manifest.json \
  --context-filter analysis \
  --verify-source-file-hashes \
  --output /mnt/kld/candidate-analysis-kld.json
```

The result contains:

- token-level mean, median, p95, p99, p99.9, and maximum KLD;
- Jensen–Shannon divergence and top-1 agreement;
- source-cluster bootstrap intervals, stratified by allocation with fixed
  allocation weights;
- micro token average and macro allocation-stratum average;
- allocation-stratum and semantic-class estimates;
- context-depth buckets;
- the 20 highest-KLD and 20 most top-1-discordant source contexts;
- comparator numerical settings and input identities.

Compare two candidate receipts with paired source-cluster statistics:

```bash
python models/kimi-k3/tools/compare-kimi-k3-fidelity-receipts.py \
  --candidate-a-report /mnt/kld/candidate-a-analysis-kld.json \
  --candidate-b-report /mnt/kld/candidate-b-analysis-kld.json \
  --candidate-a-label candidate-a \
  --candidate-b-label candidate-b \
  --output /mnt/kld/candidate-a-vs-candidate-b.json
```

Negative `difference_a_minus_b` values favor candidate A. The paired receipt
reports mean and median context differences, source-cluster bootstrap
confidence intervals stratified by allocation, per-stratum differences, win
counts, and the largest context-level disagreements.

## Hidden-state replay qualification

The replay path was checked against the 32-context live-logit qualification
suite identified in `validation/hidden-replay-qualification.json`, using the
official MXFP4 checkpoint and the canonical LM head.

| Check | Result |
|---|---:|
| Mean `KL(live logits || replayed logits)` | `1.229325e-6` |
| Maximum per-token replay KLD | `0.00195` |
| p99.9 replay KLD | `0.000274` |
| Top-1 agreement | `0.999954` |
| Alternative vocabulary-chunk result delta | `1.49e-9` |

The replay discrepancy is substantially below the candidate KLD differences
that this dataset is designed to resolve.

## Runtime-repeat control

The official MXFP4 runtime was captured three times on the same 64 sentinel
contexts. Pairwise offline replay results were:

The 64 contexts represent 57 `(dataset, source_cluster_id)` sampling units.

| Repeat pair | Mean KLD | Source-cluster bootstrap 95% CI | Top-1 agreement |
|---|---:|---:|---:|
| 00 versus 01 | `0.0032166686` | `[0.00269235, 0.00379022]` | `0.98326056` |
| 00 versus 02 | `0.0031814546` | `[0.00267033, 0.00371788]` | `0.98338269` |
| 01 versus 02 | `0.0031337795` | `[0.00261258, 0.00368719]` | `0.98348192` |

This variation is part of the measurement uncertainty. A candidate difference
comparable to sentinel repeat variation requires repeated candidate capture and
paired confidence intervals; a single point estimate is not sufficient.

The live-logit sentinels distinguish transformer/runtime variation from an
offline LM-head replay defect. Hidden-state replay and live-logit comparison
produced identical metrics for all three repeat pairs under the frozen
comparator.

## Interpretation

Use the analysis partition to reject corrupt kernels and tune quantization.
Freeze codec parameters, layer assignments, and acceptance thresholds before
opening qualification results. Report both micro and macro estimates, paired
confidence intervals, sentinel repeat variation, class-specific regressions,
and the exact runtime manifest.

Do not assign a universal quality label from a KLD threshold measured on a
different model, corpus, tokenizer, vocabulary, or serving runtime. KLD ranks
Kimi K3 candidates only within this artifact's frozen identities and does not
replace coding, reasoning, long-context, multimodal, tool-use, or free-running
generation evaluation.
