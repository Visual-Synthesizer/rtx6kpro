---
pretty_name: Kimi K3 Distribution-Fidelity Reference 1024x2048
license: other
task_categories:
  - text-generation
language:
  - en
  - zh
  - multilingual
---

# Kimi K3 distribution-fidelity reference

## Status

The artifact is **qualified** for paired, teacher-forced comparison of Kimi K3
compressed checkpoints against the official MXFP4 checkpoint. It contains
1,024 distinct 2,048-token source contexts and 2,096,128 scored next-token
positions.

The primary metric is:

```text
KL(official MXFP4 distribution || candidate distribution)
```

The reference stores BF16 hidden states after final RMSNorm and before the
language-model head. `lm-head/weight.safetensors` reconstructs the complete
163,840-token distribution through the included deterministic comparator.

## Immutable identifiers

- suite manifest SHA-256: `f3a79f7f28365d406a19a82cf210c25adf18974c4b9b607ab3754e9939f941cf`
- suite token SHA-256: `70cd72175fcb4574123b90a83e2b71bb460b8c2938cec9573645dc8c9a46a3bd`
- official checkpoint: `moonshotai/Kimi-K3` revision `2496450e92e425c886db095102a52a6682ca3970`
- canonical LM-head file SHA-256: `c282a205aed4d19fbbe4ff907feff979c50b24723e8f552ddc141ef008c5de11`
- capture runtime manifest SHA-256: `ec02e1633b045d9e7d525b41a02f20b2e1ab08adceabb4338a678b716039a783`

Run `sha256sum --check checksums.txt` after downloading. Mixed source licenses
are recorded per source in `sources.json` and `source-registry.json`; no single
license replaces those source-specific terms.

## Contents

```text
manifest.json
suite-manifest.json
capture-runtime.json
sources.json
source-registry.json
partitions.json
tokens/
reference-hidden/
lm-head/
sentinel-hidden/
sentinel-live-logits/
capture-tools/
comparators/
validation/
checksums.txt
```

Each `reference-hidden/hidden_NNNN.safetensors` file contains BF16
`hidden_states` with shape `[2047, 7168]`. The shared LM head contains BF16
`weight` with shape `[163840, 7168]`.

The 768-context analysis partition supports quantization design and parameter
selection. The 256-context qualification partition supports evaluation only
after codec parameters and acceptance thresholds are frozen. Sixty-four
analysis contexts are runtime-repeat sentinels.

## Capture a candidate

The candidate server must use the model code, TP16/DCP1 topology, B12X MLA,
Triton KDA prefill, BF16 activations and KV cache, batching configuration, and
canonical LM-head weight recorded in `capture-runtime.json`. Runtime logs must
prove the intended checkpoint, quantization path, and kernels.

`capture-tools/serve-kimi-k3-fidelity-capture.sh` requires
`CAPTURE_VLLM_SOURCE`, `CAPTURE_OUTPUT`, and a writable `CAPTURE_JIT_CACHE`.
`CAPTURE_HF_CACHE` defaults to `/root/.cache/huggingface`.

```bash
python capture-tools/capture-kimi-k3-hidden-suite.py \
  --url http://127.0.0.1:8001/v1/completions \
  --model Kimi-K3 \
  --suite-dir . \
  --capture-dir /mnt/kld/candidate-runtime/capture-hidden \
  --output-dir /mnt/kld/candidate-hidden \
  --runtime-manifest /mnt/kld/candidate-runtime.json \
  --run-name candidate-checkpoint-identifier \
  --context-filter analysis \
  --delete-raw-chunks-after-finalize
```

The client submits stored token IDs directly and resumes from an atomically
updated manifest. It does not retokenize source text.

## Compare a candidate

```bash
python comparators/compare-kimi-k3-hidden-replay.py \
  --reference-hidden-dir reference-hidden \
  --candidate-hidden-dir /mnt/kld/candidate-hidden \
  --lm-head lm-head/weight.safetensors \
  --suite-manifest suite-manifest.json \
  --context-filter analysis \
  --verify-source-file-hashes \
  --output /mnt/kld/candidate-analysis-kld.json
```

The comparator processes the complete vocabulary in bounded chunks and reports
token statistics, top-1 agreement, allocation-stratified source-cluster
bootstrap intervals, depth buckets, class estimates, and high-disagreement
source identities.

Compare two candidate result receipts with:

```bash
python comparators/compare-kimi-k3-fidelity-receipts.py \
  --candidate-a-report /mnt/kld/candidate-a-analysis-kld.json \
  --candidate-b-report /mnt/kld/candidate-b-analysis-kld.json \
  --candidate-a-label candidate-a \
  --candidate-b-label candidate-b \
  --output /mnt/kld/candidate-a-vs-candidate-b.json
```

Negative `difference_a_minus_b` values favor candidate A. The paired report
includes mean and median context differences, source-cluster win counts,
allocation-stratified confidence intervals, and the largest disagreements.

## Numerical qualification and repeat variation

Offline LM-head replay reproduced the 32-context live-logit reference with mean
KLD `1.229325e-6`, p99.9 KLD `0.000274`, maximum KLD `0.00195`, and top-1
agreement `0.999954`. Changing the vocabulary chunk size changed the aggregate
result by `1.49e-9`.

Three official-checkpoint captures over the same 64 sentinels produced pairwise
mean KLD values from `0.0031337795` to `0.0032166686`, with top-1 agreement from
`0.98326056` to `0.98348192`. The 64 sentinels represent 57 source clusters;
hidden-state replay and live-logit comparison produced identical metrics for
all three repeat pairs. Candidate improvements comparable to that repeat
variation require repeated capture and paired confidence intervals.

## Limitations

This artifact measures teacher-forced distribution fidelity at 2,048 input
tokens. It does not establish benchmark correctness, free-running generation
quality, multimodal capability, tool-execution correctness, long-context
behavior, or absence of cumulative routing drift.

The complete reproduction procedure is documented in the
[Kimi K3 distribution-fidelity runbook](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/kimi-k3/distribution-fidelity-1024x2048.md).
