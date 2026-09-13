# DeepSeek-V4.1-Flash — Jovian Judgement

The native text/Vision model
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
uses B12X GPU kernels and its embedded DSpark speculative decoder. K7 means
seven proposed tokens per verification cycle; the target model verifies them
before emission.

**Status: qualified** for TP4/DCP1 startup, text/Vision smoke, seeded sampling,
and the bounded RAM-Engram throughput tests below. SSD Engram is implemented
and was qualified in R36; R37 does not claim a repeated SSD benchmark.
Long-context cache churn and production model quality are not qualified here.

Image: `localinferencelab/vllm:jovian-judgement-community-20260913-r37`.
It has two filesystem layers. GLM, Qwen and DeepSeek V4 entrypoints are retained;
the image's default entrypoint serves GLM. DS4.1 needs its dedicated entrypoint.

## Start the server

Stage the complete checkpoint and download the Compose file:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ./models/DeepSeek-V4.1-Flash
curl -fLO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds41-jovian-judgement-r37.yml
docker compose -f docker-compose-ds41-jovian-judgement-r37.yml pull
```

Read ngram tables from local SSD:

```bash
ENGRAM_TABLE_MEMORY=disk \
docker compose -f docker-compose-ds41-jovian-judgement-r37.yml up -d
```

Keep the complete ngram tables in pinned host RAM:

```bash
ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r37.yml up -d
```

Both commands use GPUs 0–3 and listen on `0.0.0.0:8000`. The API is
`http://SERVER:8000/v1`, model name `DeepSeek-V4.1-Flash`.
For GPUs 4–7, a different port, or a checkpoint staged elsewhere:

```bash
GPU0=4 GPU1=5 GPU2=6 GPU3=7 PORT=8001 \
DS41_MODEL_DIR=./models/DeepSeek-V4.1-Flash ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r37.yml up -d
```

Check readiness with `curl -f http://localhost:8000/health` and logs with
`docker logs -f ds41-jovian-judgement`. Changing the environment and running
`up -d` recreates the service. Keep the named `ds41-cache` compilation-cache
volume. The model mount is read-only and inference is offline after staging.

## RAM versus SSD ngrams

Engram contains learned ngram lookup tables. It is not KV cache, prefix caching,
or an LMCache tier. Target and draft transformer weights remain on the GPUs.

| Setting | Placement | Host-memory requirement |
|---|---|---|
| `ENGRAM_TABLE_MEMORY=disk` (default) | Native io_uring reads selected checkpoint rows from local SSD into bounded staging buffers | Does not pin complete tables; performance depends on storage and request pattern |
| `ENGRAM_TABLE_MEMORY=ram` | Complete packed tables in pinned, GPU-mapped host RAM | Tested checkpoint: 188.83 GiB across TP4, plus loader and serving memory |

RAM mode reads the checkpoint at startup and then accesses resident tables.
Allocation failure does not silently fall back to disk. Leave memory for the
operating system and model loading. Use fast local SSD/NVMe for disk mode.

The accepted values are `ram` and `disk`, not `ssd`. The native equivalent is
`--engram-config '{"cpu_offload":false,"table_memory":"ram"}'`, or `"disk"`.
Generic CPU model offload remains disabled.

## Serving defaults

| Setting | Value |
|---|---|
| Entrypoint | `/usr/local/bin/serve-ds41-jovian.sh` |
| Parallelism | TP4/DCP1, one host; no expert-parallel CLI flag |
| Speculation | Embedded DSpark K7; greedy proposals, standard rejection, adaptive verification |
| GPU backends | B12X attention, MoE and linear; B12X PCIe communication with NCCL fallback |
| Scheduler / context | 4,096 tokens, four maximum sequences, 131,072-token context |
| Sampling | Temperature 1.0, top-p 0.95; explicit request values override defaults |
| Reasoning | `high=75`; `low=50`, `max=100`; compatibility alias `xhigh=75` |
| Decode graphs | `FULL_AND_PIECEWISE`, native target and draft captures up to 128 |
| 4,096-token prefill graphs | Off; optional `VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| Host / NCCL | OMP8, 16 channels, 2 MiB buffers |
| Loading | InstantTensor with file-backed tensor metadata |
| External KV | LMCache is not enabled or qualified for this DS4.1 profile |

Although the CLI selects `--kv-cache-dtype fp8`, DS4.1's native cache is
heterogeneous: MXFP8 sliding-window payloads, NVFP4 indexed payloads, and
index/state groups. It is not an all-FP8 cache. Expert weights use FP4 and the
execution path follows the model's W4A8 contract.

Optional prefill capture retains graph segments around dynamic attention,
not a single full-prefill graph. A same-process paired test measured +1.29%
at 32K; the integrated capture profile reduced KV capacity by approximately
11% relative to its control. It is therefore opt-in. Ordinary target/draft
decode graphs remain enabled with prefill capture off.

Named reasoning budgets follow the
[publisher's encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py).
An explicit numeric budget remains available, for example
`"chat_template_kwargs":{"reasoning_effort":50}` in a chat request.
Budget changes can alter reasoning length and speculative acceptance.

### Environment forwarding

The wrapper preserves general `VLLM_*` and `B12X_*` settings, including
`VLLM_LOGGING_LEVEL`. For debug logs through the supplied Compose:

```bash
VLLM_LOGGING_LEVEL=DEBUG \
docker compose -f docker-compose-ds41-jovian-judgement-r37.yml up -d
```

Only explicitly listed GLM tuning values inherited from the common image are
removed. Different-value overrides survive. Docker does not distinguish an
explicit caller value identical to an image default, so a listed exact value
is treated as inherited. The native DS4.1 script separately owns its fixed
architecture, thread and communication settings. Compose must explicitly pass
any additional host environment variables into the container.

## Qualification and measurements

The [R37 receipt](deepseek-v4.1-flash/r37/qualification.json) identifies the
tested image and source trees. Hardware: the same physical quartet of
4× RTX PRO 6000 Blackwell Workstation GPUs, VRAM offset **+6000**, graphics
offset zero, 600 W limits. These are not stock-clock figures. TP4/DCP1,
RAM Engram, DSpark K7/adaptive verification, batch 4096, temperature 1/top-p .95.

| Measurement | R37 result |
|---|---:|
| Uncached 32K prefill before / after decode | 21,399 / 21,452 tok/s |
| C1 output, default budget 75, median of three cells | 258.79 tok/s (256.83–278.65) |
| C1 verifier, default budget 75 | 104.11 steps/s |
| Draft proposal rate, default budget 75 | 728.53 tok/s |
| C1 output, explicit budget 50, median of three cells | 281.08 tok/s (268.02–282.16) |
| C1 verifier / draft proposal rate, explicit budget 50 | 110.80 steps/s / 775.36 tok/s |
| Sieve, default budget 75, median of five | 393.58 tok/s (371.94–442.91) |
| Sieve, explicit budget 50, median of five | 416.72 tok/s (385.40–439.01) |
| GPU KV pool | 1,741,084 tokens |
| Actual captured-graph memory | 1.21 GiB per rank |

Proposal rate includes rejected tokens over the complete serving loop; it is
not standalone draft-model speed or emitted output throughput. C1 output
depends on acceptance: default-budget cells emitted 2.702, 2.457 and 2.486
tokens per target step. Capacity is selected at startup, not guaranteed on
another host.

### Public-source comparison

The control uses public JJ `b40673cd` and B12X `323107ff` over the R36
dependency runtime. It is **not** a literal rerun of the published R36 image.
Control and R37 use the same physical quartet and benchmark protocol.
The 32K input bypasses chat templating; Sieve explicitly retains budget 50
because R37 corrects the named `high` default from 50 to 75.

| Measurement | Public-source control | R37 | Change |
|---|---:|---:|---:|
| Uncached 32K prefill | 14,896.51 tok/s | 21,399.18 tok/s | +43.65% |
| C1 output, budget 50, three-cell median | 238.80 tok/s | 281.08 tok/s | +17.70% |
| C1 verifier, budget 50, three-cell median | 99.00 steps/s | 110.80 steps/s | +11.92% |
| Sieve, budget 50, five-run median | 385.24 tok/s | 416.72 tok/s | +8.17% |

These are integrated observations, not separate gains attributable to every
PR. Sequential boots and stochastic sampling do not isolate thermal/order or
acceptance effects. The R36 RAM/SSD observations remain in the
[historical R36 release](deepseek-v4.1-flash/r36/release.md); its C1 client
ignored EOS, whereas these controls respect EOS. Do not mix those C1 numbers.

C1 at budget 75 and 50 ran sequentially, not as an interleaved budget A/B.
Their difference is not evidence that changing the budget alone causes a
particular speed change. Raw samples for both budgets are retained.

### Method and validation boundary

Prefill uses one excluded warmup, six unique 32,768-token measured prompts
before decode and three afterward. Each streams one token and reports
`32768 / TTFT`; counters require 32,768 locally computed tokens and zero local
or external cache hits. Input IDs are eight nonce bytes mapped to `1000+byte`,
then `1400+i%127` to the exact length. This cyclic input does not represent
every natural-text Engram I/O pattern.

C1/context-zero uses three independent cells, each with 15-second warmup and
30-second measurement, temperature 1, top-p .95 and normal EOS handling.
Context-zero means no added context, not an empty chat template. The client is
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench/blob/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py):

```bash
curl -fL https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py -o llm_decode_bench.py
uv run --with httpx --with rich llm_decode_bench.py \
  --host SERVER --port 8000 --model DeepSeek-V4.1-Flash \
  --display-mode plain --no-hw-monitor --no-resume --respect-eos \
  --token-targeting exact --skip-prefill --concurrency 1 --contexts 0 \
  --duration 30 --max-tokens 8192 --temperature 1 \
  --decode-warmup-seconds 15 --output decode-c1.json
```

Decline the optional self-update to retain the pinned client. Sieve uses
`Write a Python script that implements the Sieve of Eratosthenes.`, temperature
1/top-p .95, at most 2,000 output tokens, an excluded warmup and five requests
per budget. Speed includes reasoning and answer tokens, excluding TTFT.
All ten measured responses stopped normally; generated programs were not
executed, so this is not a coding-correctness evaluation.

Text arithmetic, a pigeon-image request and a first seeded sampling request
pass on the image. Recipe CPU tests: 206 passed. Built-image B12X CPU checks:
29 passed. The vLLM CPU subset has 43 passes, 14 CUDA skips and one graph
fixture failure reproduced identically on immutable R36 with CUDA hidden.
Native binaries and committed source trees pass the artifact audit.
Other model entrypoints, SSD throughput and long-context LMCache behavior
were not requalified by these DS4.1 tests.

## Release changes: R36 to R37

- Removes repeated CPU argument-list expansion, mutable-schema inspection,
  scalar dtype allocations and tensor-equality calls during kernel dispatch.
  No model quantization or sampling approximation is introduced.
- Adds measured B12X projection tiles, four FP32 projection partials, visible
  index-score tiles and compact mHC prefill configuration.
- Removes redundant attention-output copies and bounds short-index scratch.
- Includes the B12X small-tile barrier correction (#363) and TP3 prefill-head
  support from master `05d2c43b`; TP3 is not serving-qualified here.
- Adds optional 4096-token prefill graph capture; leaves it disabled to retain
  KV capacity. Target/draft decode graphs remain enabled.
- Corrects named reasoning budgets and preserves general VLLM/B12X environment
  options instead of clearing both namespaces.
- Retains #740 seeded-sampler warmup, RAM/SSD Engram, common model entrypoints,
  native libraries and the two-layer image layout.

## Source provenance

The [registry receipt](deepseek-v4.1-flash/r37/registry.json) records the immutable
DockerHub digest, verifies pull-by-digest against the tested image ID and
confirms exactly two layers. The [source lock](deepseek-v4.1-flash/r37/source.lock)
records component commits, trees and dependency input/output hashes.

- vLLM: JJ `b40673cd` plus #740/#743/#742/#744; composed commit `c687594b8a8`.
- B12X: master `05d2c43b` plus #364/#367/#366/#365; composed commit `f1c4e9dd5b1`.
- Both repositories publish `release/ds41-optimized-r37` as the build ref.
- LMCache: unchanged `29bc5a2e`, disabled in this DS4.1 serving profile.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`, tested revision
  `fb2764a5cf321eaa5070ca8f9e892818f477c16d`. Operators supply a checkpoint
  directory; the launcher does not force a model revision.
- Recipe: [Docker #34](https://github.com/local-inference-lab/blackwell-llm-docker/pull/34).
- Merge checklist: [vLLM #745](https://github.com/local-inference-lab/vllm/issues/745).

Dependency Python patches are hash-locked in the recipe, not hidden vLLM edits.
Schema enumeration credits yingru's existing
[PyTorch #195110](https://github.com/pytorch/pytorch/pull/195110); immutable
mutation metadata and CuTe sentinel identity are separate patches.
PyTorch main already avoids full default expansion during mutation tracking
through Jason Ansel's [#186175](https://github.com/pytorch/pytorch/pull/186175),
landed as [aea557660](https://github.com/pytorch/pytorch/commit/aea557660abe6db6a5fd249cc7353adaf36d57f4).
Its registration method passes 16 CPU value/version-counter parity cases
against R37; a complete PyTorch upgrade is not qualified by that check.
The CuTe identity correction is submitted as
[NVIDIA CUTLASS #3634](https://github.com/NVIDIA/cutlass/pull/3634), with seven
host-only regression methods and explicit manual-review/AI attribution.
These upstream references do not change R37's pinned dependency versions.
Fourteen native binaries are unchanged from R36.

**Public source parity:** JJ `9342b1ae809` and B12X master `fd3c638c` include
all listed vLLM/B12X PRs. Their complete Git trees match the respective R37
image source trees exactly. The #743 merge preserves its authored commit and
both ownership/indexer and prefill-capture test groups; 15 focused checks pass.
Building equivalent vLLM/B12X sources alone does not install the PyTorch/CuTe
dependency patches. The source-locked Docker recipe is still required for
runtime equivalence. LMCache remains unchanged.
