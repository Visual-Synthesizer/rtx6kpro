# DeepSeek-V4.1-Flash — Jovian Judgement

This page describes the native text and Vision model
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
with the B12X GPU kernel stack and DSpark speculative decoding in the shared
community Docker image. K7 means seven proposed tokens per verification cycle;
the target model checks them before they can be emitted.

**Status: qualified** for TP4/DCP1 startup, seeded sampling, text/Vision smoke,
and the bounded throughput tests below, with both RAM and SSD Engram storage.
This is not a production-quality or long-context cache-churn qualification.

Image: `localinferencelab/vllm:jovian-judgement-community-20260912-r36`.
The image has two filesystem layers. DS4.1 uses its own entrypoint; the default
image entrypoint still serves GLM.

## Start the server

Stage the complete checkpoint on local storage and download the Compose file:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ./models/DeepSeek-V4.1-Flash
curl -fLO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds41-jovian-judgement-r36.yml
docker compose -f docker-compose-ds41-jovian-judgement-r36.yml pull
```

Read ngram tables from SSD:

```bash
ENGRAM_TABLE_MEMORY=disk \
docker compose -f docker-compose-ds41-jovian-judgement-r36.yml up -d
```

Keep complete ngram tables in host RAM instead:

```bash
ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r36.yml up -d
```

Both commands use GPUs 0–3 and listen on `0.0.0.0:8000`; the OpenAI-compatible
endpoint is `http://SERVER:8000/v1`, model name `DeepSeek-V4.1-Flash`.
Changing the environment and running `up -d` recreates the service.

For GPUs 4–7, another port, or an already staged checkpoint:

```bash
GPU0=4 GPU1=5 GPU2=6 GPU3=7 PORT=8001 \
DS41_MODEL_DIR=./models/DeepSeek-V4.1-Flash \
ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r36.yml up -d
```

Check readiness and logs:

```bash
curl -f http://localhost:8000/health
docker logs -f ds41-jovian-judgement
```

The first launch compiles GPU kernels and captures graphs. Keep the named
`ds41-cache` volume between starts. The checkpoint bind is read-only and
inference is configured offline after staging.

## RAM versus SSD ngrams

Engram is the model's learned ngram lookup storage, not KV cache, prefix
caching, or an LMCache tier. The target and draft transformer weights remain
on the GPUs in either mode.

| Setting | Table placement | Host-memory behavior |
|---|---|---|
| `ENGRAM_TABLE_MEMORY=disk` (default) | Checkpoint files on local SSD | Native io_uring reads selected rows into bounded staging buffers; it does not pin the entire table set. |
| `ENGRAM_TABLE_MEMORY=ram` | Complete packed tables in pinned, GPU-mapped host RAM | The tested checkpoint allocates 188.83 GiB across TP4, in addition to loader and serving memory. |

RAM mode reads the checkpoint during startup, then accesses the resident
tables through the mapped host allocation. It does not silently fall back to
disk if allocation fails. Leave adequate host memory for model loading and the
operating system; 188.83 GiB is the table allocation, not total host usage.

Disk mode still needs host staging memory. Put the checkpoint on a fast local
SSD/NVMe device, not a network filesystem. Its observed latency depends on the
storage device and workload. No OS page-cache flushing is required to run it.

The public values are `ram` and `disk`, not `ssd`. The native CLI equivalent is
`--engram-config '{"cpu_offload":false,"table_memory":"ram"}'` or the same
JSON with `"disk"`. Generic CPU model offload remains disabled.

## Serving profile

| Setting | Value |
|---|---|
| Model-specific entrypoint | `/usr/local/bin/serve-ds41-jovian.sh` |
| Parallelism | Tensor Parallelism 4 (TP4) / Decode Context Parallelism 1 (DCP1), one host; no expert-parallel CLI flag |
| Drafting | Native DSpark weights contained in the target checkpoint, K7 |
| Proposal / rejection | Greedy proposals, standard rejection, adaptive verification |
| Backends | B12X attention, mixture-of-experts (MoE) and linear; B12X PCIe communication with NCCL fallback |
| Scheduler | 4,096 tokens; four maximum sequences |
| Context limit | 131,072 tokens |
| Sampling defaults | Temperature 1.0, top-p 0.95; explicit request settings override them |
| Graph configuration | `FULL_AND_PIECEWISE`, native target/draft graph capture up to 128 |
| Host threads / NCCL | OMP8; 16 channels; 2 MiB buffers |
| Weight loading | InstantTensor with native file-backed tensor metadata |
| External KV storage | LMCache is not enabled or qualified by this DS4.1 profile |

The CLI selects `--kv-cache-dtype fp8`, but the model's native key/value (KV) cache is
heterogeneous: it includes microscaling 8-bit floating-point (MXFP8)
sliding-window and NVIDIA 4-bit floating-point (NVFP4) indexed payloads plus
index/state groups. This is not an all-FP8 cache. Model expert weights use
4-bit floating-point (FP4); the execution path follows DS4.1's 4-bit-weight,
8-bit-activation (W4A8) contract.

The DS4.1 entrypoint removes GLM-specific inherited VLLM/B12X tuning and invokes
the bundled native `serve-ds41-flash.sh`. It does not copy GLM's speculative
head or KDA settings into DS4.1. The runtime uses the V2 model runner, with
completed full target/draft and piecewise graph captures in the qualification
logs. The 4,096-token prefill is not claimed to be a single full CUDA graph.

## Qualification and measurements

The [qualification receipt](deepseek-v4.1-flash/r36/qualification.json) includes
numeric samples, cache-hit counters and source identities. The workload is TP4 RTX PRO 6000
Blackwell Workstation, VRAM offset +6000, graphics offset zero, 600 W limits,
temperature 1 and top-p 0.95. These are not stock-clock figures.

Prefill uses unique uncached 32,768-token prompts, one output token, an excluded
warmup and a 30-second measured window. C1/context-zero uses three warmed
30-second cells; Sieve uses one excluded warmup and five measured requests.
Context-zero means no added context; the benchmark's short chat prompt still
contains 91 tokens.
Sieve counts reasoning and answer tokens; it is a coding-prompt speed probe,
not an evaluation of generated program correctness.

Both modes ran sequentially on the same physical GPUs 4–7, RAM first. No
sampling seeds were paired between the speed tests. C1 uses the server's
top-p 0.95 default; the prefill and Sieve requests send it explicitly.

| Engram placement | 32K prefill before / after decode (tok/s) | C1 output median (min–max tok/s) | Sieve median, 5 runs (min–max tok/s) |
|---|---:|---:|---:|
| RAM | 14,279 / 13,988 | 260.8 (247.0–299.2) | 391.7 (384.1–417.5) |
| SSD | 14,062 / 14,030 | 244.3 (237.6–245.7) | 361.4 (342.1–381.2) |

The observed RAM medians exceed SSD by **6.7% in C1 output** and **8.4% in
Sieve output**. These are storage-mode observations, not an R35-to-R36 or
sampler-patch speedup. Prefill is approximately 14K tok/s in both modes; the
synthetic cyclic-token input does not represent every natural-text ngram I/O
pattern. The RAM before/after difference is −2.0%, and both windows are retained.

| Engram placement | C1 verifier steps/s | Draft proposal tokens/s | Emitted tokens per target step | GPU KV pool tokens |
|---|---:|---:|---:|---:|
| RAM | 97.35 | 681.43 | 2.679 | 1,648,493 |
| SSD | 90.43 | 633.04 | 2.702 | 1,666,402 |

The proposal rate includes rejected tokens and is measured over the complete
serving loop. It is not standalone draft-model speed or final generated
throughput. Verifier rate and acceptance are reported separately because both
affect output tok/s. Adaptive verification also changes work per target step.
The KV allocator chooses capacity at startup; these pools are observations,
not fixed allocations promised on another host.

Both modes passed a first seeded temperature-1/top-p-0.95 request, arithmetic
and pigeon-image smoke requests, target/draft graph capture, and all measured
requests without an engine failure. The exact image passed **106 GPU tests**
for sampler parity and warmup. The source-locked recipe passed **200 CPU tests**.

### Benchmark client

The C1 and Sieve source is
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench/blob/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py).
For a C1/context-zero cell, use:

```bash
curl -fL https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py -o llm_decode_bench.py
uv run --with httpx --with rich llm_decode_bench.py \
  --host SERVER --port 8000 --model DeepSeek-V4.1-Flash \
  --display-mode plain --no-hw-monitor --no-resume \
  --token-targeting exact --skip-prefill --concurrency 1 --contexts 0 \
  --duration 30 --max-tokens 8192 --temperature 1 \
  --decode-warmup-seconds 15 --output decode-c1.json
```

Decline the optional self-update prompt when reproducing this pinned version.
Repeat with three distinct output paths and take their median. For the coding
probe, use `--coding-peak --coding-peak-runs 5 --coding-peak-temperature 1
--coding-peak-max-tokens 2000` after one separate unrecorded warmup request.
The prompt is `Write a Python script that implements the Sieve of Eratosthenes.`
Do not override the server's top-p default during these comparisons.

The exact prefill probe submits token IDs directly: eight nonce bytes mapped
to token IDs `1000 + byte`, followed by `1400 + i % 127` until the prompt has
32,768 tokens. It streams one output token at temperature 1/top-p 0.95 and
reports `32768 / TTFT`. Each receipt requires 32,768 locally computed tokens
and zero local or external prefix-cache hits. This avoids treating cache
replay as prefill or using an approximate character-to-token count.

## Release changes: R35 to R36

- Adds the native DeepSeek-V4.1 text/Vision and DSpark K7 serving profile,
  portable Compose, and explicit RAM/SSD Engram placement.
- Updates vLLM to Jovian Judgement with
  [#738](https://github.com/local-inference-lab/vllm/pull/738) merged. The greedy
  Markov sampler fuses addition and full-vocabulary reduction; incompatible
  head widths are rejected before GPU access.
- Includes [#740](https://github.com/local-inference-lab/vllm/pull/740), the
  separate startup correction for seeded processed-FP32 target sampling.
  This prevents a first-request JIT failure without changing sampling math.
- Uses unmodified B12X master `323107ff`, including its DS4.1 native kernels;
  CUDA, PyTorch, FlashInfer, FlashKDA and compatible native libraries are
  retained. Liburing is included for disk table reads.
- Preserves the common GLM, Qwen and DeepSeek V4 entrypoints and the two-layer
  image layout. This release qualification is for DS4.1, not a fresh performance
  qualification of every other model in the image.

R35 did not qualify DS4.1, so there is no matched R35-to-R36 DS4.1 speedup claim.
The isolated #738 K7 reduction measured 135.82 → 13.85 microseconds, about
0.122 ms saved and 89.8% lower latency, with identical token selection. That is
not a 9.8× model speedup. Whole-model A/B observations depend on acceptance and
adaptive verifier width; a stable whole-model gain from #738 is not established.

An intermittent within-process prefill decrease was observed in the DS4.1
test stack and remains unexplained. Successful repeat measurements do not
prove its cause fixed. Long-context cache churn, LMCache and sustained
production-quality evaluation are outside this release's bounded smoke and
throughput qualification.

## Source provenance

Immutable image:

```text
localinferencelab/vllm@sha256:23ab683d7ce32083f33c163df7dfe554b59aba75bbae8f5b8e4b5a2ef590209b
```

The [registry receipt](deepseek-v4.1-flash/r36/registry.json) verifies that
pulling this digest returns the tested image, source-lock label and two
filesystem layers. The [source lock](deepseek-v4.1-flash/r36/source.lock)
records full component and input hashes.

The image contains complete Git sources and a source-locked manifest at
`/opt/glm53-flash/source.lock`; it has no serving-source bind mounts.

- vLLM: merged JJ `b40673cd006` plus the startup correction in #740;
  installed commit `202a11a98eb`.
- B12X: clean master `323107ff`.
- LMCache: retained `29bc5a2e`; disabled for this serving profile.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash`, tested revision
  `fb2764a5cf321eaa5070ca8f9e892818f477c16d`; all 48 downloaded weight-shard
  metadata records identify that revision. The launch recipe uses the model
  directory supplied by the operator, not a forced model revision.
- Docker recipe: [blackwell-llm-docker #32](https://github.com/local-inference-lab/blackwell-llm-docker/pull/32).

Image inclusion does not mean that #740 or any LMCache PR is merged upstream.
