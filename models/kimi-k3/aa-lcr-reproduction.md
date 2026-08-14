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
| Official MXFP4 no-spec result | unsupported | A complete set of 300 generation and equality-checker receipts is not recorded on this page. |
| QSRT-K2 no-spec result | unsupported | A complete set of 300 generation and equality-checker receipts is not recorded on this page. |
| DSpark operational result | unsupported | A complete set of 300 generation and equality-checker receipts is not recorded on this page. |

The word *qualified* on this page means that all 100 questions have three
generation receipts, all 300 answers have equality-checker receipts, and every
artifact hash closes against the pinned manifests.

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
configuration. Results on this page use the version 4.1.1 equality checker and
must not be compared as though the two graders were identical.

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

The official MXFP4 and QSRT-K2 no-spec servers must use identical values for:

- vLLM and B12X source revisions;
- TP and DCP topology;
- activation and KV-cache dtypes;
- attention and KDA backends;
- maximum model length and prefill chunk size;
- prefix-caching policy;
- CUDA graph policy;
- tokenizer and chat template;
- request order and sampling parameters.

Checkpoint identity and checkpoint-required quantization kernels are the only
intended differences. Prefix caching may be enabled because questions sharing
a document set have an identical document prefix. Requests remain in CSV order
so both checkpoints receive the same cache opportunity.

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
    "decode_context_parallel_size": 8
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
    "max_num_seqs": 1,
    "kv_cache_memory_bytes": 1860000000,
    "prefix_caching": true,
    "compilation_config": {
      "cudagraph_mode": "PIECEWISE",
      "cudagraph_capture_sizes": [1]
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
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 200000 \
  --timeout-seconds 7200
```

Each receipt preserves the full API response, final answer, reasoning content
when returned by the server, usage counters, finish reason, elapsed time,
ordered document hashes, prompt hash, and generation-configuration hash.

## Equality checker

Set `OPENAI_API_KEY` in the process environment without placing it in shell
history or an artifact. The judge uses no explicit temperature unless the
provider requires one.

```bash
python3 models/kimi-k3/tools/run-kimi-k3-aa-lcr.py judge \
  --dataset-root "$AA_LCR_ROOT" \
  --generation-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR" \
  --base-url https://api.openai.com/v1 \
  --model gpt-5.6-luna \
  --api-key-env OPENAI_API_KEY \
  --reasoning-effort medium \
  --max-tokens 4096 \
  --timeout-seconds 600
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
  --judge-dir "$RUN_DIR" \
  --output "$RUN_DIR/summary.json"
```

`summary.json` is marked `qualified` only when it contains 300 judge receipts,
question IDs 1 through 100, repeats 0 through 2, and three receipts per
question. Partial summaries are marked `research-only`.

## Artifact layout

```text
<run-id>/
  runtime-manifest.json
  docker-inspect.json
  generation-manifest.json
  judge-manifest.json
  responses/
    repeat-00/question-0001.json
    repeat-01/question-0001.json
    repeat-02/question-0001.json
    ...
  judgements/
    repeat-00/question-0001.json
    repeat-01/question-0001.json
    repeat-02/question-0001.json
    ...
  summary.json
```

## Result table

| Checkpoint | Speculation | TP/DCP | Pass@1 | Correct / attempts | Generation receipt hash | Judge receipt hash | Status |
|---|---:|---:|---:|---:|---|---|---|
| Official MXFP4 `2496450e…` | disabled | unrecorded | — | — | — | — | unsupported |
| QSRT-K2 `3b981141…` | disabled | unrecorded | — | — | — | — | unsupported |
| QSRT-K2 `3b981141…` | Inferact DSpark | unrecorded | — | — | — | — | unsupported |

## Interpretation limits

AA-LCR measures open-answer reasoning over document sets of approximately 72K
to 115K `cl100k_base` tokens. It does not measure short-context coding,
multimodal behavior, tool execution, 1M-token retrieval, decode throughput, or
teacher-forced distribution fidelity. KLD, throughput, and AA-LCR therefore
remain separate qualification dimensions.
