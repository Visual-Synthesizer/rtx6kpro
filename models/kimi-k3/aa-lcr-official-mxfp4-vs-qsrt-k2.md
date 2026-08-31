# Kimi-K3 AA-LCR official MXFP4 versus QSRT K2

## Purpose and status

This page specifies the qualified paired AA-LCR comparison between:

- the official `moonshotai/Kimi-K3` MXFP4 checkpoint at revision
  `2496450e92e425c886db095102a52a6682ca3970`; and
- the `lukealonso/Kimi-K3-QSRT-K2` routed-expert checkpoint at revision
  `3b98114115f1d41ce7963ba346c3fca19918b0bd`.

Both checkpoints were evaluated without speculative decoding. Each checkpoint
generated three independent answers for every one of the 100 questions in the
pinned Artificial Analysis Long Context Reasoning dataset. Two equality
checkers scored the same 600 answers.

Status: **qualified**. Every expected generation and judgement receipt is
present, every hash closes, every API response used for scoring finished
without an error, and both paired comparison receipts pass structural
validation.

## Results

| Equality checker | Official MXFP4 | QSRT K2 | QSRT K2 minus official |
|---|---:|---:|---:|
| Frozen official Kimi-K3 | 254/300 (84.67%) | 245/300 (81.67%) | -3.00 percentage points |
| GPT-5.6 Sol, maximum reasoning | 249/300 (83.00%) | 237/300 (79.00%) | -4.00 percentage points |

Per-repeat correct counts were:

| Checkpoint | Frozen official Kimi-K3 | GPT-5.6 Sol |
|---|---|---|
| Official MXFP4 | 86, 84, 84 | 86, 81, 82 |
| QSRT K2 | 84, 79, 82 | 81, 77, 79 |

The frozen Kimi-K3 judge produced 236 both-correct pairs, 18 official-only
correct pairs, 9 QSRT-only correct pairs, and 37 both-incorrect pairs. Its
exact two-sided McNemar p-value is `0.1220781207`. A 200,000-replicate
question-cluster bootstrap produced a 95% interval of `[-0.0600, -0.0033]` for
the QSRT-minus-official score difference.

The GPT-5.6 Sol judge produced 230 both-correct pairs, 19 official-only correct
pairs, 7 QSRT-only correct pairs, and 44 both-incorrect pairs. Its exact
two-sided McNemar p-value is `0.0289592743`. The corresponding
question-cluster bootstrap interval is `[-0.0767, -0.0067]`.

The exact paired test and the question-cluster bootstrap answer different
questions. McNemar's test uses the 26 or 27 discordant binary attempt labels.
The bootstrap resamples 100 questions while keeping the three observed
generations and judge labels fixed. It does not include additional model or
judge repeat variation.

The two equality checkers agreed on 295 of 300 official-MXFP4 labels (98.33%)
and 292 of 300 QSRT-K2 labels (97.33%). Agreement establishes that both judge
families found the same broad result; it does not make either judge an official
Artificial Analysis score.

Repository receipts:

- [frozen official Kimi-K3 paired comparison](validation/aa-lcr-official-mxfp4-vs-qsrt-k2-k3-judge-20260815.json);
- [GPT-5.6 Sol paired comparison](validation/aa-lcr-official-mxfp4-vs-qsrt-k2-sol-judge-20260815.json);
- [evidence archive identity](validation/aa-lcr-official-mxfp4-vs-qsrt-k2-archive-20260815.json).

## Immutable evidence

The complete artifact is stored in the Hugging Face dataset repository
[`festr2/kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2`](https://huggingface.co/datasets/festr2/kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2/tree/4bbc2b3b17314b83cf330e3c7a7d4c10d32daff5)
at revision `4bbc2b3b17314b83cf330e3c7a7d4c10d32daff5`.

| Object | Identity |
|---|---|
| Archive | `kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2-20260815.tar.gz` |
| Archive size | 1,987,454 bytes |
| Archive SHA-256 | `1a5ebe1adfc1249af1e9ebc1b49693346203c07ba5defb4dcb0bc9b16eb70ecc` |
| Checksummed files | 3,145 |
| Generation receipts | 600 |
| Judgement receipts | 1,300 |
| Frozen Kimi-K3 paired comparison | `paired-frozen-official-kimi-k3.json` |
| GPT-5.6 Sol paired comparison | `paired-gpt-5.6-sol-max.json` |

The 1,300 judgement receipts comprise 600 frozen Kimi-K3 labels, 600 GPT-5.6
Sol labels, and 100 frozen Kimi-K3 repeatability-control labels. The archive
also contains all raw generation API responses, judge logs, runtime manifests,
launchers, comparison utilities, and the deterministic archive packager.

Checkpoint weights and AA-LCR source documents are not redistributed. Raw
`docker inspect` output is also omitted because a process environment can
contain credentials. The runtime manifests preserve image digests, source
revisions, checkpoint revisions, argument vectors, relevant non-secret
environment variables, and hashes of the omitted inspection records.

## Download and verify the evidence

```bash
ARTIFACT_DIR=/srv/kimi-k3-aa-lcr-evidence
ARTIFACT_REVISION=4bbc2b3b17314b83cf330e3c7a7d4c10d32daff5
ARCHIVE=kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2-20260815.tar.gz

mkdir -p "$ARTIFACT_DIR"
hf download \
  festr2/kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2 \
  "$ARCHIVE" archive-receipt.json \
  --repo-type dataset \
  --revision "$ARTIFACT_REVISION" \
  --local-dir "$ARTIFACT_DIR"

printf '%s  %s\n' \
  1a5ebe1adfc1249af1e9ebc1b49693346203c07ba5defb4dcb0bc9b16eb70ecc \
  "$ARTIFACT_DIR/$ARCHIVE" | sha256sum --check

tar -xzf "$ARTIFACT_DIR/$ARCHIVE" -C "$ARTIFACT_DIR"
EVIDENCE_ROOT="$ARTIFACT_DIR/kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2-20260815"
(cd "$EVIDENCE_ROOT" && sha256sum --check checksums.sha256)
```

## Immutable dataset inputs

| Object | Identity |
|---|---|
| Repository | `ArtificialAnalysis/AA-LCR` |
| Revision | `bdae010bbce259820c0e34c1d7cce210d966fb75` |
| Question CSV SHA-256 | `2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea` |
| Extracted-document ZIP SHA-256 | `5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a` |
| Prompt-manifest SHA-256 | `13f8fdc097679d5ead0c4bba6044b254a1fcd80f8e5afb9555c68bd3d0abd09d` |
| Kimi-K3 token-count manifest SHA-256 | `ca980972df40cf1dacde770f9e2f80fe9e4c4ab74c5c72407c70669a5fcf54de` |

Prepare and validate the source data:

```bash
AA_LCR_ROOT=/srv/aa-lcr-bdae010

git clone --no-tags \
  https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR \
  "$AA_LCR_ROOT"
git -C "$AA_LCR_ROOT" checkout --detach \
  bdae010bbce259820c0e34c1d7cce210d966fb75
git -C "$AA_LCR_ROOT" lfs pull \
  --include='extracted_text/AA-LCR_extracted-text.zip'

mkdir -p "$AA_LCR_ROOT/extracted_text/unpacked"
unzip -q -n \
  "$AA_LCR_ROOT/extracted_text/AA-LCR_extracted-text.zip" \
  -d "$AA_LCR_ROOT/extracted_text/unpacked"

python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" validate \
  --dataset-root "$AA_LCR_ROOT"

python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" token-counts \
  --dataset-root "$AA_LCR_ROOT" \
  --tokenizer moonshotai/Kimi-K3 \
  --tokenizer-revision 2496450e92e425c886db095102a52a6682ca3970 \
  --output "$AA_LCR_ROOT/kimi-k3-token-counts.json"

sha256sum "$AA_LCR_ROOT/kimi-k3-token-counts.json"
```

The last command must report
`ca980972df40cf1dacde770f9e2f80fe9e4c4ab74c5c72407c70669a5fcf54de`.
The pinned Kimi-K3 tokenizer produces 71,136 to 114,776 input tokens per
request, with a median of 95,119.5.

## Generation contract

Every request contains one user message and no system message. Documents are
inserted in the CSV order. Both checkpoint runs use:

```text
repeats = 3
reasoning_effort = max
temperature = 1.0
top_p = 0.95
max_tokens = 200000
request seed = omitted
system message = absent
streaming = disabled
client concurrency = 16
```

`max_tokens` is a ceiling. All 600 qualified responses ended with `stop`.
Prefix caching and speculative decoding were disabled.

The server configurations differ because the checkpoint storage formats have
different memory footprints:

| Property | Official MXFP4 | QSRT K2 |
|---|---:|---:|
| Tensor parallel size | 16 | 16 |
| Decode context parallel size | 16 | 8 |
| FP8 physical KV tokens | 1,060,357 | 1,102,812 |
| Maximum model length | 1,048,576 | 1,048,576 |
| Scheduler token budget | 2,048 | 4,096 |
| Maximum active sequences | 8 | 8 |
| Attention / KDA prefill | B12X MLA / Triton | B12X MLA / Triton |
| MoE / linear backend | B12X / B12X | B12X / B12X |
| Weight loader | InstantTensor | InstantTensor |

AA-LCR therefore qualifies each checkpoint plus its named serving
configuration. It does not isolate routed-expert quantization error. The
teacher-forced distribution-fidelity suite performs that controlled isolation.

## Reconstruct the serving source and image

The qualified runs used container digest
`voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971`.
The image ID recorded by both runtime manifests is
`sha256:f226a6fd788bb4af345a17b768654f1e5a7487a812746ccb117aa9b040a82294`.

Prepare the source checkouts and apply the archived working-tree patch:

```bash
VLLM_SOURCE=/srv/kimi-k3-vllm-c203914
B12X_SOURCE=/srv/kimi-k3-b12x-f9f6fd4

git clone https://github.com/local-inference-lab/vllm.git "$VLLM_SOURCE"
git -C "$VLLM_SOURCE" checkout --detach \
  c203914d1b146032ed8a788f37037c3d835fc684
git -C "$VLLM_SOURCE" apply \
  "$EVIDENCE_ROOT/runs/official-mxfp4/runtime-artifacts/vllm-working-tree.patch"

git clone https://github.com/local-inference-lab/b12x.git "$B12X_SOURCE"
git -C "$B12X_SOURCE" checkout --detach \
  f9f6fd4ad4d82ed7bf3a3523689f0b230a46eb0d

mkdir -p /root/vllm/kimi/source-overlay
cp \
  "$EVIDENCE_ROOT/runs/official-mxfp4/runtime-artifacts/source-overlay-sitecustomize.py" \
  /root/vllm/kimi/source-overlay/sitecustomize.py

docker pull \
  voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971
```

Download the model snapshots at their immutable revisions. The commands are
large downloads and require sufficient local storage:

```bash
OFFICIAL_MODEL=/srv/models/Kimi-K3-2496450
QSRT_MODEL=/srv/models/Kimi-K3-QSRT-K2-3b98114

hf download moonshotai/Kimi-K3 \
  --revision 2496450e92e425c886db095102a52a6682ca3970 \
  --local-dir "$OFFICIAL_MODEL"
hf download lukealonso/Kimi-K3-QSRT-K2 \
  --revision 3b98114115f1d41ce7963ba346c3fca19918b0bd \
  --local-dir "$QSRT_MODEL"
```

The archive contains the exact launchers. Set paths for the machine executing
the reproduction; do not edit the archived copies.

Official MXFP4:

```bash
IMAGE=voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971 \
MODEL="$OFFICIAL_MODEL" \
VLLM_SOURCE="$VLLM_SOURCE" \
B12X_SOURCE="$B12X_SOURCE" \
FLASH_ATTN_FORWARD="$EVIDENCE_ROOT/runs/official-mxfp4/runtime-artifacts/flash-fwd.py" \
PORT=8001 \
CACHE=/srv/kimi-k3-cache/official-aa-lcr \
CONTAINER=kimi-k3-official-mxfp4-aa-lcr \
bash "$EVIDENCE_ROOT/runs/official-mxfp4/runtime-artifacts/launch-server.sh"
```

QSRT K2:

```bash
IMAGE=voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971 \
MODEL="$QSRT_MODEL" \
VLLM_SOURCE="$VLLM_SOURCE" \
B12X_SOURCE="$B12X_SOURCE" \
FLASH_ATTN_FORWARD="$EVIDENCE_ROOT/runs/official-mxfp4/runtime-artifacts/flash-fwd.py" \
PORT=8001 \
KV_CACHE_MEMORY_BYTES=1950000000 \
MAX_NUM_BATCHED_TOKENS=4096 \
MAX_NUM_SEQS=8 \
SERVED_MODEL_NAME=Kimi-K3-QSRT-K2-NoSpec-TP16-DCP8-1M-CC8-MBT4096 \
CACHE=/srv/kimi-k3-cache/qsrt-k2-aa-lcr \
CONTAINER=kimi-k3-qsrt-k2-aa-lcr \
bash "$EVIDENCE_ROOT/runs/qsrt-k2/runtime-artifacts/delegated-launch-server.sh"
```

Only one of these launch commands can occupy the 16 GPUs at a time. The
official checkpoint must be served again when it acts as the frozen Kimi-K3
equality checker.

## Generate and seal answers

Create a `runtime-manifest.json` for each serving execution before generation.
The archived files under `runs/*/runtime-manifest.json` are field-complete
examples. A reproduction manifest must record the executing container ID,
image ID and registry digest, checkpoint index hash, source commits and patch
hash, full server argument vector, topology, dtypes, backend selectors, KV
allocation, graph policy, and relevant non-secret environment variables. Do
not copy an archived container identity into a different execution.

For either server, set `RUN_DIR` and `SERVED_MODEL` to the corresponding
checkpoint identity:

```bash
python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" generate \
  --dataset-root "$AA_LCR_ROOT" \
  --base-url http://127.0.0.1:8001/v1 \
  --model "$SERVED_MODEL" \
  --output-dir "$RUN_DIR" \
  --runtime-manifest "$RUN_DIR/runtime-manifest.json" \
  --repeats 3 \
  --concurrency 16 \
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 200000 \
  --timeout-seconds 7200

python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" verify-generations \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --token-count-manifest "$AA_LCR_ROOT/kimi-k3-token-counts.json" \
  --output "$RUN_DIR/generation-completeness.json"
```

The generation command is resumable. It skips only a receipt whose prompt and
generation-configuration hashes match. An incompatible receipt stops the run
instead of being overwritten.

## Frozen official Kimi-K3 equality checker

Serve the official MXFP4 configuration on port 8001. For each generation
directory, run:

```bash
K3_JUDGE_DIR="$RUN_DIR/judges/frozen-official-kimi-k3-max-temp0"

python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" judge \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --output-dir "$K3_JUDGE_DIR" \
  --base-url http://127.0.0.1:8001/v1 \
  --model Kimi-K3-Official-MXFP4-NoSpec-TP16-DCP16-1M-CC8-FA2 \
  --judge-protocol frozen-official-kimi-k3 \
  --judge-runtime-manifest "$OFFICIAL_RUN_DIR/runtime-manifest.json" \
  --reasoning-effort max \
  --temperature 0 \
  --max-tokens 32768 \
  --timeout-seconds 1800 \
  --concurrency 16

python3 "$EVIDENCE_ROOT/tools/run-kimi-k3-aa-lcr.py" summarize \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --judge-dir "$K3_JUDGE_DIR" \
  --output "$K3_JUDGE_DIR/pass-at-1-summary.json"
```

The judge receives only the question, official answer, and candidate answer.
It uses `temperature=0`, maximum reasoning effort, no seed, no system message,
and a 32,768-token output ceiling. The official Kimi-K3 judge is common to
both checkpoints but is not externally independent.

## GPT-5.6 Sol equality-checker control

The independent control uses Codex CLI `0.147.0`, model `gpt-5.6-sol`, maximum
reasoning effort, an ephemeral session, an empty read-only workspace, and no
user configuration. Authenticate Codex with ChatGPT before running the
control:

```bash
codex login status

SOL_JUDGE_DIR="$RUN_DIR/judges/gpt-5.6-sol-max-codex-chatgpt"

python3 "$EVIDENCE_ROOT/tools/judge-kimi-k3-aa-lcr-codex.py" \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --output-dir "$SOL_JUDGE_DIR" \
  --reference-judge-dir "$K3_JUDGE_DIR" \
  --codex-bin codex \
  --model gpt-5.6-sol \
  --reasoning-effort max \
  --concurrency 4 \
  --timeout-seconds 600
```

GPT-5.6 Sol is an independent control, not the GPT-5.6 Luna medium-reasoning
judge named by Artificial Analysis methodology version 4.1.1. The Codex model
alias and account service are not immutable model-weight revisions, so exact
future replay is not guaranteed even when the CLI arguments match. The
archived raw outputs and receipt hashes preserve the qualified execution.

## Paired comparison

Run one comparison for each equality-checker family. The command requires 300
qualified receipts in each directory and rejects mismatched judge contracts.

```bash
python3 "$EVIDENCE_ROOT/tools/compare-kimi-k3-aa-lcr-scores.py" \
  --reference-dir "$OFFICIAL_JUDGE_DIR" \
  --candidate-dir "$QSRT_JUDGE_DIR" \
  --reference-label official-mxfp4 \
  --candidate-label qsrt-k2 \
  --bootstrap-replicates 200000 \
  --bootstrap-seed 20260815 \
  --output paired-official-mxfp4-vs-qsrt-k2.json
```

The qualified comparison receipts are available individually in the pinned
Hugging Face revision and inside the archive under `comparisons/`.

## Rebuild the deterministic evidence archive

The packager rejects an existing output path, excludes raw Docker inspection
records, normalizes tar metadata, writes a per-file checksum manifest, and
stores the archive checksum in a sidecar receipt.

```bash
python3 "$EVIDENCE_ROOT/tools/package-kimi-k3-aa-lcr-evidence.py" \
  --official-run "$OFFICIAL_RUN_DIR" \
  --candidate-run "$QSRT_RUN_DIR" \
  --candidate-name qsrt-k2 \
  --k3-comparison paired-official-mxfp4-vs-qsrt-k2-k3.json \
  --sol-comparison paired-official-mxfp4-vs-qsrt-k2-sol.json \
  --token-count-manifest "$AA_LCR_ROOT/kimi-k3-token-counts.json" \
  --tools-dir "$EVIDENCE_ROOT/tools" \
  --output kimi-k3-aa-lcr-official-mxfp4-vs-qsrt-k2.tar.gz
```

## Interpretation limits

This evaluation measures free-running, open-answer reasoning over document
sets that tokenize to approximately 71K through 115K Kimi-K3 input tokens. It
does not measure short-context coding, multimodal behavior, tool execution,
one-million-token retrieval, serving throughput, or teacher-forced
distribution fidelity. Three generations per question do not characterize the
full sampling distribution, and one judge pass per generated answer does not
measure judge-repeat variance.
