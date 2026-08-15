# Kimi K3 AA-LCR reproduction specification

## Purpose

This page specifies a reproducible comparison of Kimi K3 checkpoints on the
[Artificial Analysis Long Context Reasoning dataset](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR).
The primary comparison measures the official `moonshotai/Kimi-K3` MXFP4
checkpoint and `lukealonso/Kimi-K3-QSRT-K2` without speculative decoding.
Inferact DSpark results are reported separately because speculative decoding is
an inference configuration rather than a checkpoint property.

## Qualification status

| Component | Status | Evidence |
|---|---|---|
| Dataset identity and document resolution | implemented | The pinned CSV, ZIP, document paths, and generated prompts pass `run-kimi-k3-aa-lcr.py validate`. |
| Generation and receipt harness | implemented | `models/kimi-k3/tools/run-kimi-k3-aa-lcr.py` writes one atomic raw-response receipt per question and repeat. |
| TP16/DCP16 hybrid-cache prefix caching | unsupported | Three identical greedy 94,557-token requests produced two output hashes and two preemptions; see [prefix-cache qualification](validation/aa-lcr-prefix-cache-tp16-dcp16-20260814.json). |
| Official MXFP4 no-spec generation | qualified | All 100 questions have three hash-verified receipts and all 300 responses finished with `stop`; see [generation completeness](validation/aa-lcr-official-mxfp4-nospec-generation-tp16-dcp16-20260814.json) and [execution evidence](validation/aa-lcr-official-mxfp4-nospec-execution-tp16-dcp16-20260814.json). |
| Frozen official Kimi-K3 equality-checker protocol | qualified | Two independent executions agreed on 100 of 100 fixed-answer labels; see [repeatability evidence](validation/aa-lcr-k3-judge-repeatability-20260814.json). |
| Official MXFP4 no-spec score with the frozen Kimi-K3 judge | qualified | The judge marked 254 of 300 attempts correct, or 84.67%; see [score evidence](validation/aa-lcr-official-mxfp4-nospec-k3-judge-tp16-dcp16-20260814.json). |
| Official MXFP4 no-spec score with GPT-5.6 Sol maximum reasoning | qualified | The independent control marked 249 of 300 attempts correct, or 83.00%; complete receipts are in the [paired evidence artifact](aa-lcr-official-mxfp4-vs-qsrt-k2.md). |
| Official MXFP4 no-spec score with the Artificial Analysis GPT-5.6 Luna judge | unsupported | No GPT-5.6 Luna equality-checker receipts are recorded. |
| QSRT-K2 no-spec generation | qualified | All 100 questions have three hash-verified receipts and all 300 responses finished with `stop`; see [generation completeness](validation/aa-lcr-qsrt-k2-nospec-generation-tp16-dcp8-20260815.json). |
| QSRT-K2 no-spec score with the frozen Kimi-K3 judge | qualified | The judge marked 245 of 300 attempts correct, or 81.67%; complete receipts are in the [paired evidence artifact](aa-lcr-official-mxfp4-vs-qsrt-k2.md). |
| QSRT-K2 no-spec score with GPT-5.6 Sol maximum reasoning | qualified | The independent control marked 237 of 300 attempts correct, or 79.00%; complete receipts are in the [paired evidence artifact](aa-lcr-official-mxfp4-vs-qsrt-k2.md). |
| DSpark operational result | unsupported | A complete set of 300 generation and equality-checker receipts is not recorded on this page. |

The word *qualified* for a generation artifact means that all 100 questions
have three receipts and every artifact hash closes against the pinned
manifests. A qualified score additionally requires 300 valid equality-checker
receipts under one explicitly named judge protocol. Judge protocols define
different result families and their absolute scores are not interchangeable.

## Immutable inputs

| Object | Identity |
|---|---|
| Question and document repository | `ArtificialAnalysis/AA-LCR` |
| Git revision | `bdae010bbce259820c0e34c1d7cce210d966fb75` |
| Question CSV SHA-256 | `2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea` |
| Extracted-document ZIP SHA-256 | `5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a` |
| Prompt-manifest SHA-256 | `13f8fdc097679d5ead0c4bba6044b254a1fcd80f8e5afb9555c68bd3d0abd09d` |
| Kimi K3 token-count manifest SHA-256 | `ca980972df40cf1dacde770f9e2f80fe9e4c4ab74c5c72407c70669a5fcf54de` |
| Questions | 100 |
| Document sets | 30 |
| Referenced documents | 229 |
| Documents present in the ZIP | 230 |

The dataset README reports 234 documents, but the pinned CSV references 229
unique paths and the pinned ZIP contains 230 text files. One ZIP member is not
referenced by the CSV:

```text
Legal/legal_eu_ai/Preparing for change_ How businesses can thrive under the EU_s AI Act _ Global law firm _ Norton Rose Fulbright.txt
```

The CSV spelling `Başev` and the ZIP spelling `Başev` differ only in Unicode
composition. The harness resolves filenames after NFC normalization and does
not rename source files. Two question-to-document references require this
normalization.

The pinned Kimi K3 tokenizer and chat template produce 71,136 to 114,776 input
tokens per request, with a median of 95,119.5. The token-count receipt is
`kimi-k3-token-counts.json`; it records every question ID and both raw-prompt
and chat-formatted token counts.

## Methodology identity

The evaluation follows [Artificial Analysis Intelligence Benchmarking version
4.1.1](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
published in August 2026:

- 100 open-answer questions;
- three independent samples per question;
- pass@1 is mean correctness over all 300 samples;
- the equality checker is GPT-5.6 Luna with medium reasoning effort;
- the equality checker must return only `CORRECT` or `INCORRECT`.

The pinned dataset README names Qwen3 235B A22B 2507 Non-reasoning as the
equality checker. That statement identifies the dataset-card scoring
configuration. The frozen Kimi-K3 result recorded on this page uses neither
Qwen3 nor GPT-5.6 Luna and is not an official Artificial Analysis result.
Scores produced by different equality checkers must not be compared as though
the graders were identical.

Kimi K3 is a reasoning model. Its pinned model card overrides the generic
Artificial Analysis reasoning-model temperature:

```text
reasoning_effort = max
temperature = 1.0
top_p = 0.95
max_tokens = 200000
request seed = omitted
system message = absent
```

The same sampling parameters apply to official MXFP4 and QSRT-K2. The
`max_tokens` value is a ceiling, not a forced output length. Each server must
expose at least 128K context; the qualified server profiles use the native
1,048,576-token Kimi K3 limit.

## Frozen official Kimi-K3 judge protocol

The internal paired-comparison protocol uses the official
`moonshotai/Kimi-K3` checkpoint at revision
`2496450e92e425c886db095102a52a6682ca3970` as one common equality checker for
every candidate checkpoint and inference configuration. The serving-runtime
manifest has SHA-256
`80ef33848eedd6a123648698864d439e4f65ce2b57f575ec220f35982968a34e`.
The judge request configuration is:

```text
reasoning_effort = max
temperature = 0
max_tokens = 32768
system message = absent
request seed = omitted
```

The judge receives only the question, official answer, and candidate answer.
It does not receive the candidate checkpoint identity. The protocol is useful
for paired checkpoint comparisons because every candidate is evaluated by the
same frozen model and prompt. It does not provide an externally independent
absolute score: an official Kimi-K3 judge can share error preferences with
Kimi-K3 candidates.

The qualified 300-attempt execution consumed 33 to 5,519 completion tokens per
judge request. A 4,096-token judge limit would have truncated at least one
request, so 32,768 is the required ceiling for this protocol.

Repeatability was measured by judging the same 100 fixed candidate answers in
two independent executions. Binary labels and final answer text agreed on
100 of 100 pairs. Internal reasoning text agreed bit-for-bit on 29 pairs and
completion-token counts agreed on 34 pairs. The classification was stable;
the reasoning trace was not deterministic. The durable measurements are in
[the repeatability receipt](validation/aa-lcr-k3-judge-repeatability-20260814.json).

## Dataset preparation

```bash
AA_LCR_ROOT=/mnt/luke/evals/aa-lcr-bdae010

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

python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py validate \
  --dataset-root "$AA_LCR_ROOT"

python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py token-counts \
  --dataset-root "$AA_LCR_ROOT" \
  --tokenizer moonshotai/Kimi-K3 \
  --tokenizer-revision 2496450e92e425c886db095102a52a6682ca3970 \
  --output "$AA_LCR_ROOT/kimi-k3-token-counts.json"
```

The validator fails on a changed CSV or ZIP hash, an absent question ID, an
NFC filename collision, a missing referenced document, or a changed number of
questions, document sets, or referenced documents.

## Prompt construction

Every request contains exactly one user message and no system message. Files
are inserted in the order listed by the CSV `data_source_filenames` field.

```text
BEGIN INPUT DOCUMENTS

BEGIN DOCUMENT 1:
{document 1 text}
END DOCUMENT 1

BEGIN DOCUMENT 2:
{document 2 text}
END DOCUMENT 2

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
```

The harness hashes every completed prompt. The aggregate prompt-manifest hash
listed under **Immutable inputs** identifies all 100 prompts and their order.

## Server comparison contract

AA-LCR qualifies a complete checkpoint-and-serving configuration. Compared
configurations must use identical dataset inputs, prompt construction,
tokenizer and chat template, generation sampling, repeat count, and equality
checker. The serving topology, cache dtype, kernel selection, scheduler budget,
and source revisions may differ when a checkpoint requires or benefits from a
different runtime. Every such difference must be present in the runtime
manifest and result table.

Consequently, an AA-LCR score can establish operational quality for the named
runtime but cannot isolate checkpoint quantization error. The distribution-
fidelity suite in `distribution-fidelity-1024x2048.md` provides the controlled
checkpoint comparison: it uses matched model code, topology, activation dtype,
attention backend, batching, and one shared LM head for the reference and
candidate captures.

Prefix caching must be disabled for the qualified TP16/DCP16 hybrid MLA-Mamba
server profile. The qualification in
`validation/aa-lcr-prefix-cache-tp16-dcp16-20260814.json` reduced cached-request
latency from 40.85 seconds to approximately 4.63 seconds, but identical greedy
requests produced different output hashes and the server recorded two
preemptions. The latency reduction is therefore not valid evaluation evidence.

DSpark uses a separate run identifier and result table. It must not replace a
no-spec checkpoint result.

## Generate answers

The command is resumable. A completed compatible receipt is skipped; a receipt
with a different prompt or generation-configuration hash stops the run.

Before generation, create `runtime-manifest.json` from the serving container.
The manifest is part of the generation identity and must contain enough
information to reconstruct the server without relying on a mutable image tag:

```json
{
  "status": "qualified",
  "checkpoint": {
    "repository": "moonshotai/Kimi-K3",
    "revision": "2496450e92e425c886db095102a52a6682ca3970",
    "index_sha256": "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
  },
  "container": {
    "image": "repository/name:immutable-tag",
    "image_id": "sha256:...",
    "registry_digest": "repository/name@sha256:..."
  },
  "source": {
    "vllm_revision": "full Git commit",
    "b12x_revision": "full Git commit"
  },
  "topology": {
    "tensor_parallel_size": 16,
    "decode_context_parallel_size": 16
  },
  "serving": {
    "activation_dtype": "bfloat16",
    "kv_cache_dtype": "fp8",
    "attention_backend": "B12X_MLA",
    "kda_prefill_backend": "triton",
    "moe_backend": "b12x",
    "linear_backend": "b12x",
    "weight_loader": "instanttensor",
    "max_model_len": 1048576,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 8,
    "kv_cache_memory_bytes": 960000000,
    "prefix_caching": false,
    "compilation_config": {
      "cudagraph_mode": "FULL_DECODE_ONLY",
      "cudagraph_capture_sizes": [1, 8]
    }
  },
  "server_arguments": ["vllm", "serve", "..."],
  "relevant_environment": {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
```

Record the complete `docker inspect` output next to the concise manifest. The
concise manifest must identify the checkpoint index hash, image ID and registry
digest, source commits, exact server argument vector, topology, dtypes,
backends, graph policy, KV allocation, and environment variables that select
runtime code paths.

```bash
AA_LCR_ROOT=/mnt/luke/evals/aa-lcr-bdae010
RUN_DIR=/mnt/luke/evals/kimi-k3-aa-lcr/<run-id>

python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py generate \
  --dataset-root "$AA_LCR_ROOT" \
  --base-url http://127.0.0.1:8000/v1 \
  --model '<served-model-name>' \
  --output-dir "$RUN_DIR" \
  --runtime-manifest "$RUN_DIR/runtime-manifest.json" \
  --repeats 3 \
  --concurrency 16 \
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 200000 \
  --timeout-seconds 7200
```

`--concurrency` controls the number of independent HTTP requests kept in
flight. It does not change the server's `max_num_seqs` limit. The value is part
of the generation-configuration hash, so resuming an output directory with a
different client concurrency is rejected. Each response is written atomically
to its own question-and-repeat receipt before another task is reported as
complete.

`--repeat-scheduling question_serial` is implemented for server profiles that
have independently qualified prefix caching. It assigns all repeats of one
question to one client worker and submits them in repeat order. The qualified
TP16/DCP16 Kimi K3 profile omits this option and independently schedules every
question-repeat pair because its prefix-caching path is unsupported.

Each receipt preserves the full API response, final answer, reasoning content
when returned by the server, usage counters, finish reason, elapsed time,
ordered document hashes, prompt hash, and generation-configuration hash.

## Verify generation completeness

The `verify-generations` command reconstructs every prompt and ordered document
list from the pinned dataset, verifies the pinned Kimi K3 prompt-token counts,
checks the runtime and generation-configuration hashes, verifies each candidate
answer hash, and requires exactly three `stop` receipts for every question.

```bash
uv run --no-project --with requests -- python \
  models/kimi-k3/tools/run-kimi-k3-aa-lcr.py verify-generations \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --output "$RUN_DIR/generation-completeness.json"
```

The official MXFP4 no-spec generation artifact is stored at:

```text
/mnt/luke/evals/kimi-k3-aa-lcr/official-mxfp4-nospec-tp16-dcp16-aa-lcr-ws4096-20260814
```

Its serving runtime has these identities and properties:

- checkpoint `moonshotai/Kimi-K3` revision
  `2496450e92e425c886db095102a52a6682ca3970`;
- full official MXFP4 routed experts, BF16 dense tensors and activations, and no
  speculative decoder;
- TP16, DCP16, FP8 KV cache, 1,060,357 reported physical KV tokens, and
  `max_model_len=1,048,576`;
- B12X MLA attention, FlashAttention 2 MLA prefill, Triton KDA prefill, B12X
  MoE and linear backends, and InstantTensor weight loading;
- prefix caching disabled, server `max_num_seqs=8`, client concurrency 16, and
  prefill workspace 4,096 tokens;
- container image ID
  `sha256:f226a6fd788bb4af345a17b768654f1e5a7487a812746ccb117aa9b040a82294`
  and registry digest
  `voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971`;
- vLLM revision `c203914d1b146032ed8a788f37037c3d835fc684` with working-tree
  patch SHA-256
  `54f5d2fb6692e65d61cf05b6c086c40bc383f4e6d568d168f4586fc63fe6a363`;
- B12X revision `f9f6fd4ad4d82ed7bf3a3523689f0b230a46eb0d`.

The qualified generation evidence contains 300 receipts, 100 for each repeat,
with zero failure sidecars and 300 `stop` finish reasons. The model consumed
28,549,305 prompt tokens and produced 349,532 completion tokens. Completion
length was 149 to 14,349 tokens, with median 646, p95 3,378.05, and p99
8,303.12. Per-request elapsed time was 220.76 to 3,021.07 seconds; elapsed
times overlap because requests were concurrent and must not be summed as wall
time or used as a throughput measurement.

The serving process recorded one request preemption, zero container restarts,
no CUDA out-of-memory failure, and zero failure receipts. The resumed client
execution wrote 285 receipts, preserved 15 compatible receipts, and exited with
status 0. The generation completeness receipt has SHA-256
`fc3f0a27d36d61934755a012cb46b2f70217af5b6eb4126d488a40e11528afd9`;
the canonical manifest over all response paths and file hashes is
`1e32bd3415f7279feda510061944115b0bf2560ef025bdd19757433feb7edda0`.

This evidence qualifies generation integrity. Correctness under the frozen
official Kimi-K3 judge is qualified by a separate set of 300 equality-checker
receipts. Correctness under the Artificial Analysis GPT-5.6 Luna judge remains
unsupported.

The QSRT-K2 target-only generation artifact is stored at:

```text
/mnt/luke/evals/kimi-k3-aa-lcr/qsrt-k2-nospec-tp16-dcp8-aa-lcr-cc8-mbt4096-20260814
```

It uses TP16/DCP8, FP8 KV cache, 1,102,812 reported physical KV tokens,
`max_model_len=1,048,576`, a 4,096-token scheduler budget, eight active
sequences, B12X MLA/MoE/linear kernels, Triton KDA prefill, and InstantTensor.
The runtime manifest has SHA-256
`5de71f080b3e363cd42ebcc7f113e1d36fb5a78a50fa111234fd786cd8a4cc80`.

The qualified generation evidence contains 300 `stop` receipts, zero failure
sidecars, and 28,549,305 prompt tokens. The model produced 343,933 completion
tokens: 125 to 13,062 tokens per request, with median 665, p95 3,357.95, and
p99 4,778.19. Per-request elapsed time was 163.52 to 4,370.02 seconds. The
canonical response-file manifest has SHA-256
`e3c4002728c1a99b88dd1a8d995e87792c3c229b3e2086743a38e7e5dba8b154`.
These values qualify generation integrity, not answer correctness or serving
throughput.

## Equality checker

The frozen Kimi-K3 judge writes receipts below a dedicated judge directory so
that its result cannot be confused with an Artificial Analysis judge result:

```bash
GENERATION_DIR=/mnt/luke/evals/kimi-k3-aa-lcr/official-mxfp4-nospec-tp16-dcp16-aa-lcr-ws4096-20260814
JUDGE_DIR="$GENERATION_DIR/judges/frozen-official-kimi-k3-max-temp0-20260814"

python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py judge \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$JUDGE_DIR" \
  --base-url http://127.0.0.1:8001/v1 \
  --model Kimi-K3-Official-MXFP4-NoSpec-TP16-DCP16-1M-CC8-FA2 \
  --judge-protocol frozen-official-kimi-k3 \
  --judge-runtime-manifest "$GENERATION_DIR/runtime-manifest.json" \
  --reasoning-effort max \
  --temperature 0 \
  --max-tokens 32768 \
  --timeout-seconds 1800 \
  --concurrency 16
```

Compatible receipts are skipped on a resumed execution. A failure writes an
atomic `question-NNNN.error.json` sidecar without replacing a qualified
receipt. `--start-question`, `--stop-question`, and `--repeat` select bounded
subsets for diagnostics without changing judge identity.

The Artificial Analysis version 4.1.1 judge requires a separate output
directory and an OpenAI API credential. Set `OPENAI_API_KEY` in the process
environment without placing it in shell history or an artifact:

```bash
AA_JUDGE_DIR="$GENERATION_DIR/judges/artificial-analysis-v4.1.1-gpt-5.6-luna"

python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py judge \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$GENERATION_DIR" \
  --output-dir "$AA_JUDGE_DIR" \
  --base-url https://api.openai.com/v1 \
  --model gpt-5.6-luna \
  --judge-protocol artificial-analysis-v4.1.1 \
  --api-key-env OPENAI_API_KEY \
  --reasoning-effort medium \
  --max-tokens 32768 \
  --timeout-seconds 1800 \
  --concurrency 16
```

The equality-checker prompt is:

```text
Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT.
```

Any other judge label is an error and is not coerced into a score.

## Summarize receipts

```bash
python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py summarize \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$GENERATION_DIR" \
  --judge-dir "$JUDGE_DIR" \
  --output "$JUDGE_DIR/pass-at-1-summary.json"
```

The summary is marked `qualified` only after revalidating all 300
question-repeat pairs, generation-receipt hashes, judge-prompt hashes, response
model identities, labels, correctness flags, and `stop` finish reasons. Any
failure sidecar or partial receipt set produces `research-only` status.

## Artifact layout

```text
<run-id>/
  runtime-manifest.json
  generation-manifest.json
  generation-completeness.json
  generation-execution-receipt.json
  runtime-artifacts/
    docker-inspect.json
    launch-server.sh
    vllm-working-tree.patch
  responses/
    repeat-00/question-0001.json
    repeat-01/question-0001.json
    repeat-02/question-0001.json
    ...
  judges/
    <judge-protocol-id>/
      judge-manifest.json
      judgements/
        repeat-00/question-0001.json
        repeat-01/question-0001.json
        repeat-02/question-0001.json
        ...
      pass-at-1-summary.json
```

## Result table

| Checkpoint | Speculation | TP/DCP | Judge | Pass@1 | Correct / attempts | Generation receipt hash | Judge receipt hash | Status |
|---|---:|---:|---|---:|---:|---|---|---|
| Official MXFP4 `2496450e…` | disabled | TP16/DCP16 | Frozen official Kimi-K3 `2496450e…` | 84.67% | 254 / 300 | `1e32bd3415…` | `ba4b2d2e28…` | qualified |
| Official MXFP4 `2496450e…` | disabled | TP16/DCP16 | GPT-5.6 Sol, maximum reasoning | 83.00% | 249 / 300 | `1e32bd3415…` | `64c590f772…` | qualified |
| Official MXFP4 `2496450e…` | disabled | TP16/DCP16 | GPT-5.6 Luna, AA v4.1.1 | — | — | `1e32bd3415…` | — | unsupported |
| QSRT-K2 `3b981141…` | disabled | TP16/DCP8 | Frozen official Kimi-K3 `2496450e…` | 81.67% | 245 / 300 | `e3c4002728…` | `e5f3b68c40…` | qualified |
| QSRT-K2 `3b981141…` | disabled | TP16/DCP8 | GPT-5.6 Sol, maximum reasoning | 79.00% | 237 / 300 | `e3c4002728…` | `bcb178eaac…` | qualified |
| QSRT-K2 `3b981141…` | Inferact DSpark | unrecorded | Frozen official Kimi-K3 `2496450e…` | — | — | — | — | unsupported |

The qualified Kimi-K3-judged summary has SHA-256
`113399528496ab1964904fb7b5acd5815455db406a25d29142b376d316e2282e`.
Its Wilson 95% interval is 80.15% to 88.30%. Per-repeat scores are 86%, 84%,
and 84%. Seventy-seven questions were correct in all three generations, nine
were incorrect in all three, and fourteen had mixed labels.

The complete paired statistics, both judge families, exact runtime identities,
downloadable receipts, and reproduction commands are specified on the
[official MXFP4 versus QSRT K2 comparison page](aa-lcr-official-mxfp4-vs-qsrt-k2.md).

## Interpretation limits

AA-LCR measures open-answer reasoning over document sets of approximately 72K
to 115K `cl100k_base` tokens. It does not measure short-context coding,
multimodal behavior, tool execution, 1M-token retrieval, decode throughput, or
teacher-forced distribution fidelity. KLD, throughput, and AA-LCR therefore
remain separate qualification dimensions.
