# GLM-5.3-Flash AA-LCR reproduction specification

## Purpose and status

This specification defines the Local Inference Lab procedure for measuring
GLM-5.3-Flash checkpoint-and-serving configurations on the Artificial Analysis
Long Context Reasoning (AA-LCR) dataset.

Status: **implemented**. The procedure produced qualified 300-attempt artifacts
for the published NVFP4 checkpoint, QAD step 1,750, and the BF16 checkpoint.
The frozen tools, configuration receipts, and aggregate validation receipts
are stored beside this document. Full candidate answers and per-attempt judge
receipts are retained by Local Inference Lab and are not distributed in the
wiki.

The corresponding results are:

- [BF16, published NVFP4, and QAD step 1,750](aa-lcr-bf16-vs-nvfp4.md); and
- [published NVFP4 versus QAD step 1,750](aa-lcr-nvfp4-vs-qad-step1750.md).

## Immutable inputs

| Object | Identity |
|---|---|
| Dataset repository | `ArtificialAnalysis/AA-LCR` |
| Dataset revision | `bdae010bbce259820c0e34c1d7cce210d966fb75` |
| Question CSV SHA-256 | `2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea` |
| Extracted-document ZIP SHA-256 | `5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a` |
| Prompt-manifest SHA-256 | `13f8fdc097679d5ead0c4bba6044b254a1fcd80f8e5afb9555c68bd3d0abd09d` |
| Referenced-document manifest SHA-256 | `969d3c7b0aba7be8cb2ccb144f5c06b022eff09307fe963953e1a5c952d7fe59` |
| Published NVFP4 checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4@378ca54585c46542bad1f3cb3ed0d73ae51cdb62` |
| Published NVFP4 index SHA-256 | `0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb` |
| QAD checkpoint | `GLM-5.3-Flash-NVFP4-QAD-step1750`, Quatrain step 1,750 |
| QAD index SHA-256 | `b43d25a280d02bfd2a58c046386e24baad78fcce355ea2d48cc0c4c78671686b` |
| BF16 checkpoint | `zai-org/GLM-5.3-Flash-BF16@61f77a1e1a67c410650ce5017411337da0dcd11a` |
| BF16 index SHA-256 | `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f` |
| Container digest | `voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340` |
| Source-lock SHA-256 | `9a6167d415d824e1707ba7df0dd5906e05c004f1ed2666f80f2f9e1ea9fde4be` |

The QAD checkpoint is not publicly downloadable. Reproducing its score requires
the exact checkpoint files identified by complete file-manifest SHA-256
`12362645b613f625f5e7bc008050db51181cb3b50fcc5e066b97a8494f9fcf33`.
The [QAD distribution-fidelity report](../../kld/glm-5.3-flash-qad-step1750.md)
specifies its contents and materialization provenance.

## Frozen tools

Run commands from the rtx6kpro repository root.

| Tool | SHA-256 |
|---|---|
| [`run-aa-lcr.py`](tools/run-aa-lcr.py) | `2925a08048300c38ea83823df311636e06e1102a75067d5ed3bf37ef388a38b7` |
| [`judge-aa-lcr-codex.py`](tools/judge-aa-lcr-codex.py) | `c4ad47d8899aceec08da1d0c9dede479491338b882c660c6dcd9578efbe4d8b3` |
| [`compare-aa-lcr-scores.py`](tools/compare-aa-lcr-scores.py) | `7339d96ec383aea19e4eb9b2fcd435a44320b2fce0f47e4b0d4c5180b0d7a6b5` |

The generator writes one atomic JSON receipt per question and repeat. Resuming
skips a receipt only after validating its prompt, documents, generation
configuration, runtime identity, response hash, and expected location.
Configuration drift stops the run. The verifier requires all 300 receipts, no
failure sidecars, exact tokenizer prompt counts, matching hashes, and only
`stop` finish reasons.

The Codex equality-checker tool starts a fresh `codex exec` session for every
answer, disables user configuration, uses an empty read-only workspace, and
accepts only one normalized binary label. It is resumable under the same
configuration hash.

## Prepare the dataset

~~~bash
git clone --no-tags \
  https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR \
  /srv/aa-lcr-bdae010

git -C /srv/aa-lcr-bdae010 checkout --detach \
  bdae010bbce259820c0e34c1d7cce210d966fb75

git -C /srv/aa-lcr-bdae010 lfs pull \
  --include=extracted_text/AA-LCR_extracted-text.zip

mkdir -p /srv/aa-lcr-bdae010/extracted_text/unpacked
unzip -q -n \
  /srv/aa-lcr-bdae010/extracted_text/AA-LCR_extracted-text.zip \
  -d /srv/aa-lcr-bdae010/extracted_text/unpacked

python3 models/glm-5.3-flash/tools/run-aa-lcr.py validate \
  --dataset-root /srv/aa-lcr-bdae010
~~~

The validator rejects a changed CSV or ZIP hash, missing question IDs, an
unexpected question or document count, unresolved filenames, and ambiguous
Unicode-normalized filenames. The pinned archive contains 230 files in 30
document sets; 229 files are referenced by the 100 questions.

## Construct and count prompts

Every generation request contains exactly one user message and no system
message. Referenced files retain the order in the dataset CSV. The user content
has this form:

~~~text
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
~~~

Generate tokenizer counts with the same immutable tokenizer as the evaluated
checkpoints:

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py token-counts \
  --dataset-root /srv/aa-lcr-bdae010 \
  --tokenizer local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --tokenizer-revision 378ca54585c46542bad1f3cb3ed0d73ae51cdb62 \
  --output /srv/aa-lcr-bdae010/glm-5.3-flash-token-counts.json

sha256sum /srv/aa-lcr-bdae010/glm-5.3-flash-token-counts.json
~~~

The expected SHA-256 is
`6b5b4b3fff2b3cf0179591c3ee1721474dd588dea6504031caa22fb856509562`.
The expected chat-token range is 76,820 to 114,611, with median 100,972.

## Start one NVFP4 serving replica

The following template starts one TP4/DCP1 replica. Replace angle-bracketed
values with one checkpoint path, four GPU indices, a unique port, container
name, and cache volume.

~~~bash
docker run -d \
  --name <container-name> \
  --init \
  --gpus '"device=<four-comma-separated-GPU-indices>"' \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v <checkpoint-directory>:/model:ro \
  -v <dedicated-cache-volume>:/cache \
  -e MODEL=/model \
  -e SERVED_MODEL_NAME=<served-model-name> \
  -e PORT=<port> \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=24 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=8 \
  -e CACHE_MODE=vram \
  -e KV_CACHE_QUANT=fp8_ds_mla \
  -e ENABLE_PREFIX_CACHING=1 \
  -e SPECULATOR=mtp \
  -e MTP_DEPTH=3 \
  -e GLM53_KDA_PREFILL_BACKEND=flashkda \
  -e GPU_MEMORY_UTILIZATION=0.95 \
  -e CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=256 \
  -e 'CUDAGRAPH_CAPTURE_SIZES=1 2 4 8 16 32 40 48 64 96 128 192 256' \
  voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340 \
  --kv-cache-memory-bytes 32212254720
~~~

Do not set `NCCL_GRAPH_FILE` to an empty value. The image launcher unsets it
when no real NCCL topology XML is supplied.

The qualified published-NVFP4 pool used two replicas. The qualified QAD pool
used three replicas. Every replica used identical per-replica arguments and a
dedicated JIT cache volume. Wait until `/health` succeeds, confirm the model
identity through `/v1/models`, and retain startup logs and `/metrics` before
generation.

The immutable runtime receipts show every argument, backend, source revision,
container identity, and GPU UUID:

- [published NVFP4 runtime](validation/aa-lcr-nvfp4-runtime-20260903.json);
- [QAD step 1,750 runtime](validation/aa-lcr-qad-step1750-runtime-20260903.json);
  and
- [BF16 runtime](validation/aa-lcr-bf16-runtime-20260903.json).

Create a fresh `runtime-manifest.json` for a reproduction. Do not reuse a
captured container ID. The manifest must record checkpoint and index hashes,
container image ID and registry digest, source revisions, GPU UUIDs, exact
server argument vector, topology, activation and cache dtypes, selected
backends, cache allocation, graph policy, prefix-cache state, and relevant
non-secret environment variables.

## Start the BF16 serving replica

The BF16 checkpoint requires eight 96 GiB GPUs for the qualified runtime. The
image's standard launcher selects the ModelOpt quantized path, so the BF16
configuration invokes `/opt/venv/bin/vllm` directly. Mount the complete pinned
snapshot at `/model`.

~~~bash
docker run -d \
  --name glm53-aa-lcr-bf16-mtp3 \
  --init \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v /srv/models/glm-5.3-flash-bf16-61f77a1:/model:ro \
  -v glm53-aa-lcr-bf16-cache:/cache \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_SOCKET_IFNAME=lo \
  -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_MIN_NCHANNELS=16 \
  -e NCCL_MAX_NCHANNELS=16 \
  -e NCCL_BUFFSIZE=2097152 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_DMA_MIN_BYTES=6MB \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto \
  -e VLLM_GLM53_L2_PREFETCH=1 \
  -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e 'CUDAGRAPH_CAPTURE_SIZES=1 2 4' \
  --entrypoint /opt/venv/bin/vllm \
  voipmonitor/vllm@sha256:d6ccc79f65e3b83896e7307afafc89146b2d116ef2e7166295e15bd362a5d340 \
  serve /model \
  --served-model-name GLM-5.3-Flash-BF16-MTP3-AA-LCR \
  --host 0.0.0.0 \
  --port 5054 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --cp-kv-cache-interleave-size 4 \
  --dcp-kv-cache-interleave-size 4 \
  --max-num-seqs 4 \
  --max-model-len 300000 \
  --max-num-batched-tokens 2048 \
  --prefill-schedule-interval 8 \
  --max-cudagraph-capture-size 4 \
  --kv-cache-memory 9663676416 \
  --mamba-cache-mode align \
  --enable-chunked-prefill \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format auto \
  --attention-backend B12X \
  --moe-backend auto \
  --linear-backend auto \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --additional-config \
  '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"flashkda"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"auto","attention_backend":"B12X"}'
~~~

The qualified server reports 1,272,727 physical KV-cache tokens and 4.24-way
capacity at the 300,000-token model limit. The resolved KV format is
`fp8_ds_mla`, and the resolved routed-expert implementation is FlashInfer
CUTLASS unquantized.

A 4,096-token scheduler budget combined with 0.95 automatic GPU-memory
allocation is **unsupported** for this workload. It passed a short smoke
request but failed a 95,407-token preflight prompt during a temporary B12X
multi-head-composition allocation. Use the explicit 9 GiB cache and 2,048-token
scheduler budget above. The
[failure receipt](validation/aa-lcr-bf16-bt4096-auto95-unsupported-20260903.json)
records the rejected configuration.

## Generate published-NVFP4 answers

The two `--base-url` values must address independent replicas with identical
runtime manifests. The harness assigns each question to one endpoint by
question ID modulo two and keeps its three repeats serial on that endpoint.

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py generate \
  --dataset-root /srv/aa-lcr-bdae010 \
  --base-url http://127.0.0.1:5052/v1 \
  --base-url http://127.0.0.1:5053/v1 \
  --model GLM-5.3-Flash-NVFP4-MTP3-AA-LCR \
  --output-dir /srv/glm53-aa-lcr/nvfp4 \
  --runtime-manifest /srv/glm53-aa-lcr/nvfp4/runtime-manifest.json \
  --repeats 3 \
  --concurrency-per-endpoint 24 \
  --repeat-scheduling question_serial \
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 163840 \
  --timeout-seconds 7200
~~~

Seal the generation set:

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py verify-generations \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/nvfp4 \
  --token-count-manifest /srv/aa-lcr-bdae010/glm-5.3-flash-token-counts.json \
  --output /srv/glm53-aa-lcr/nvfp4/generation-completeness.json
~~~

The qualified output identities are:

- generation manifest
  `80217f2fc90a8bd19224b081edba4e05c8dd5c98ecca2c5fa159676b4260e89c`;
- completeness receipt
  `7307c6201083b87edc1690cca26dbc76570ccbbc55e1be7d0510f29de53366f7`.

## Generate QAD step 1,750 answers

Use three identical replicas and list their base URLs. Question assignment is
modulo three; all repeats for a question remain on the assigned endpoint.

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py generate \
  --dataset-root /srv/aa-lcr-bdae010 \
  --base-url http://127.0.0.1:5051/v1 \
  --base-url http://127.0.0.1:5052/v1 \
  --base-url http://127.0.0.1:5053/v1 \
  --model GLM-5.3-Flash-NVFP4-QAD-step1750-MTP3-AA-LCR \
  --output-dir /srv/glm53-aa-lcr/qad-step1750 \
  --runtime-manifest /srv/glm53-aa-lcr/qad-step1750/runtime-manifest.json \
  --repeats 3 \
  --concurrency-per-endpoint 24 \
  --repeat-scheduling question_serial \
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 163840 \
  --timeout-seconds 7200
~~~

Seal the generation set with the same token manifest:

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py verify-generations \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/qad-step1750 \
  --token-count-manifest /srv/aa-lcr-bdae010/glm-5.3-flash-token-counts.json \
  --output /srv/glm53-aa-lcr/qad-step1750/generation-completeness.json
~~~

The qualified output identities are:

- generation manifest
  `e3c45e98b92935834533b4831783409260e87de1775958c25099308cec4a6faa`;
- completeness receipt
  `a4227af03471f8b237edfa2dccb574d680259e654fb47fc1f300638e10823e81`.

## Generate BF16 answers

Use the single qualified TP8 endpoint. Four client workers submit four question
groups; all repeats for one question remain serial in one worker.

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py generate \
  --dataset-root /srv/aa-lcr-bdae010 \
  --base-url http://127.0.0.1:5054/v1 \
  --model GLM-5.3-Flash-BF16-MTP3-AA-LCR \
  --output-dir /srv/glm53-aa-lcr/bf16 \
  --runtime-manifest /srv/glm53-aa-lcr/bf16/runtime-manifest.json \
  --repeats 3 \
  --concurrency-per-endpoint 4 \
  --repeat-scheduling question_serial \
  --reasoning-effort max \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 163840 \
  --timeout-seconds 7200
~~~

Seal the BF16 generation set with the shared token manifest:

~~~bash
python3 models/glm-5.3-flash/tools/run-aa-lcr.py verify-generations \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/bf16 \
  --token-count-manifest /srv/aa-lcr-bdae010/glm-5.3-flash-token-counts.json \
  --output /srv/glm53-aa-lcr/bf16/generation-completeness.json
~~~

The qualified BF16 output identities are:

- generation manifest
  `456822fe9b53e893ed789cdfe4f88597cb91938bdc6d29edf9216291b42288a7`;
- completeness receipt
  `de5357f136c0f888068806f08a99d85b57271e302f8cd32a59ea7ed1cb20b661`;
  and
- canonical generation receipt-set hash
  `ccfdd6e920554ac662641de8af6c4a7b212b34a28563c567c91fd6f81b4a20d1`.

## Run the equality checker

Artificial Analysis methodology version 4.1.1 names GPT-5.6 Luna at medium
reasoning as the AA-LCR equality checker. The command requires an authenticated
Codex CLI. It submits only the question, official answer, and candidate answer;
the checkpoint identity is absent from the judge prompt.

Published NVFP4:

~~~bash
python3 models/glm-5.3-flash/tools/judge-aa-lcr-codex.py \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/nvfp4 \
  --output-dir /srv/glm53-aa-lcr/nvfp4/judges/gpt-5.6-luna-medium \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --concurrency 4 \
  --timeout-seconds 600
~~~

QAD step 1,750:

~~~bash
python3 models/glm-5.3-flash/tools/judge-aa-lcr-codex.py \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/qad-step1750 \
  --output-dir /srv/glm53-aa-lcr/qad-step1750/judges/gpt-5.6-luna-medium \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --concurrency 4 \
  --timeout-seconds 600
~~~

BF16:

~~~bash
python3 models/glm-5.3-flash/tools/judge-aa-lcr-codex.py \
  --dataset-root /srv/aa-lcr-bdae010 \
  --generation-dir /srv/glm53-aa-lcr/bf16 \
  --output-dir /srv/glm53-aa-lcr/bf16/judges/gpt-5.6-luna-medium \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --concurrency 4 \
  --timeout-seconds 600
~~~

A qualified judge directory contains exactly 300 JSON receipts, no error
sidecars, a matching `judge-manifest.json`, and a
`pass-at-1-summary.json` with status `qualified`.

The judge is an external dependency whose behavior can change behind a stable
model name. A reproduction must record the model name, reasoning effort,
provider, Codex CLI version, execution-isolation settings, and execution date.
Matching the documented command does not imply bitwise reproduction of the
retained judge labels.

## Compute paired comparisons

Published NVFP4 versus QAD step 1,750:

~~~bash
python3 models/glm-5.3-flash/tools/compare-aa-lcr-scores.py \
  --reference-dir /srv/glm53-aa-lcr/nvfp4/judges/gpt-5.6-luna-medium \
  --candidate-dir /srv/glm53-aa-lcr/qad-step1750/judges/gpt-5.6-luna-medium \
  --reference-label 'Published GLM-5.3-Flash NVFP4' \
  --candidate-label 'Quatrain QAD step 1750' \
  --bootstrap-replicates 200000 \
  --bootstrap-seed 20260903 \
  --output /srv/glm53-aa-lcr/nvfp4-vs-qad-step1750.json
~~~

Published NVFP4 versus BF16:

~~~bash
python3 models/glm-5.3-flash/tools/compare-aa-lcr-scores.py \
  --reference-dir /srv/glm53-aa-lcr/nvfp4/judges/gpt-5.6-luna-medium \
  --candidate-dir /srv/glm53-aa-lcr/bf16/judges/gpt-5.6-luna-medium \
  --reference-label 'Published GLM-5.3-Flash NVFP4' \
  --candidate-label 'GLM-5.3-Flash BF16' \
  --bootstrap-replicates 200000 \
  --bootstrap-seed 20260903 \
  --output /srv/glm53-aa-lcr/nvfp4-vs-bf16.json
~~~

QAD step 1,750 versus BF16:

~~~bash
python3 models/glm-5.3-flash/tools/compare-aa-lcr-scores.py \
  --reference-dir /srv/glm53-aa-lcr/qad-step1750/judges/gpt-5.6-luna-medium \
  --candidate-dir /srv/glm53-aa-lcr/bf16/judges/gpt-5.6-luna-medium \
  --reference-label 'Quatrain QAD step 1750' \
  --candidate-label 'GLM-5.3-Flash BF16' \
  --bootstrap-replicates 200000 \
  --bootstrap-seed 20260903 \
  --output /srv/glm53-aa-lcr/qad-step1750-vs-bf16.json
~~~

The comparator requires both result sets to contain the same 300
question-repeat keys and the same equality-checker contract. It reports the
2-by-2 paired table, exact two-sided McNemar p-value, per-question difference
distribution, and a question-cluster bootstrap interval.

## Qualification boundary

A generation artifact is qualified only when:

- all 100 questions and all three repeats are present exactly once;
- dataset, referenced-document, prompt, tokenizer, generation-configuration,
  runtime, and answer hashes close;
- every server response has the pinned prompt-token count;
- every response finished with `stop`; and
- no generation failure sidecar exists.

An equality-checker artifact is qualified only when all 300 binary labels are
present, every receipt matches one judge configuration, and no judge failure
sidecar exists.

Qualification applies to the complete checkpoint-and-serving configuration.
Different tensor parallelism, cache dtype, MTP depth, prompt template,
sampling, prefix-cache policy, output ceiling, or equality checker defines a
different result. The score does not isolate quantization error; the
[GLM-5.3-Flash distribution-fidelity reports](../../kld/glm-5.3-flash-bf16-nvfp4.md)
provide the controlled teacher-relative measurement.
