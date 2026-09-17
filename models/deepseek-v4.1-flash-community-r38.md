# DeepSeek-V4.1-Flash — community R38 deployment and measurement archive

Historical record: commands and qualification below belong to community R38.
For the profile-based GHCR image, use the
[DeepSeek V4.1 model page](deepseek-v4.1-flash.md) and
[shared Docker guide](../docs/unified-vllm-docker.md).

The native text/Vision model
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
uses B12X GPU kernels and its embedded DSpark speculative decoder. K7 means
seven proposed tokens per verification cycle; the target model verifies them
before emission.

**Status: qualified** for the bounded TP4/DCP1 RAM-Engram checks below.
The measurement and correctness boundaries are stated below.

Image: `localinferencelab/vllm:jovian-judgement-community-20260914-r38`.
It has two filesystem layers. GLM, Qwen and DeepSeek V4 entrypoints are retained;
the image's default entrypoint serves GLM. DS4.1 needs its dedicated entrypoint.

## Start the server

Stage the complete checkpoint and download the Compose file:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --local-dir ./models/DeepSeek-V4.1-Flash
curl -fLO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds41-jovian-judgement-r38.yml
docker compose -f docker-compose-ds41-jovian-judgement-r38.yml pull
```

Read ngram tables from local SSD:

```bash
ENGRAM_TABLE_MEMORY=disk \
docker compose -f docker-compose-ds41-jovian-judgement-r38.yml up -d
```

Keep the complete ngram tables in pinned host RAM:

```bash
ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r38.yml up -d
```

Both commands use GPUs 0–3 and listen on `0.0.0.0:8000`. The API is
`http://SERVER:8000/v1`, model name `DeepSeek-V4.1-Flash`.
For GPUs 4–7, a different port, or a checkpoint staged elsewhere:

```bash
GPU0=4 GPU1=5 GPU2=6 GPU3=7 PORT=8001 \
DS41_MODEL_DIR=./models/DeepSeek-V4.1-Flash ENGRAM_TABLE_MEMORY=ram \
docker compose -f docker-compose-ds41-jovian-judgement-r38.yml up -d
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
Generic CPU model offload remains disabled. The Compose profile grants unlocked
memory and io_uring access required by disk staging.

Opt-in Engram controls are exposed without a default performance claim:

| Control | Behavior | Default |
|---|---|---|
| `ENGRAM_DISK_RESIDENT_SCALES` | Keep original E8M0 scale bytes in host RAM while weights stay on SSD | `0` |
| `ENGRAM_DISK_PREFETCH_MAX_TOKENS` | Bounded concurrent disk preparation up to the specified token count | `0`, disabled |
| `ENGRAM_PROJECTION_TP` | Partition Engram WKV output columns across TP ranks | `0`, disabled |

Disk component correctness is distinct from whole-model SSD throughput. R38
RAM measurements do not establish a serving speedup for these options.

## Serving defaults

| Setting | Value |
|---|---|
| Entrypoint | `/usr/local/bin/serve-ds41-jovian.sh` |
| Parallelism | TP4/DCP1, one host; no expert-parallel CLI flag |
| Speculation | Embedded DSpark K7; greedy proposals, standard rejection, adaptive verification |
| GPU backends | B12X attention, MoE and linear; B12X PCIe communication with NCCL fallback |
| Scheduler / context | 4,096 tokens, four maximum sequences, 131,072-token context |
| Main / sliding-window pages | 256 / 128 tokens; logical sliding window remains 128 |
| Fairness | Fixed prefill compute share 0.4, one parallel-prefill lane |
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
docker compose -f docker-compose-ds41-jovian-judgement-r38.yml up -d
```

Only explicitly listed GLM tuning values inherited from the common image are
removed. Different-value overrides survive. Docker does not distinguish an
explicit caller value identical to an image default, so a listed exact value
is treated as inherited. The wrapper selects DS4.1 OMP8 and capture128 instead of the inherited GLM
OMP1/capture256. An explicit `--swa-block-size` takes precedence over
`SWA_BLOCK_SIZE`. Compose must explicitly pass
any additional host environment variables into the container.

### Page geometry and adaptive verification

`BLOCK_SIZE=256` and `SWA_BLOCK_SIZE=128` reduce cache allocation overhead without
changing payload precision or the logical sliding window. Prefix-match
granularity is a separate control, not a promise to cache every token.

`DSPARK_ADAPTIVE_VERIFICATION=0` disables trimming, not DSpark itself.
`DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE=1.0` is the default; larger values
encourage more trimming. Compatible padded verification graphs are priced
instead of extrapolating a constant from a single measurement.

## Qualification and measurements

The [artifact receipt](deepseek-v4.1-flash/r38/qualification.json) and raw
[control](deepseek-v4.1-flash/r38/control/) /
[candidate](deepseek-v4.1-flash/r38/candidate/) samples record a matched
immutable-R37/R38 comparison. Hardware: the same physical quartet of four
RTX PRO 6000 Blackwell Workstation GPUs, **stock clocks**, graphics/VRAM
offsets 0, 600 W limits and driver 615.71.09. These are not historical +6000 figures.

TP4/DCP1, RAM Engram, DSpark K7/adaptive, batch 4096, max sequences 32,
context 131072, capture 128, fixed compute share 0.4, one prefill lane,
image limit 2, temperature 1, top-p .95 and numeric reasoning budget 75.
Public Compose defaults to four maximum sequences; qualification allows 32 for C8.

| Measurement | R37 control | R38 | Change |
|---|---:|---:|---:|
| Uncached 32K prefill (tok/s) | 20,950.29 | 21,035.18 | +0.41% |
| C1 output (tok/s) | 226.37 | 259.27 | +14.53% |
| C1 verifier (steps/s) | 86.87 | 101.20 | +16.49% |
| C8 aggregate output (tok/s) | 592.63 | 810.61 | +36.78% |
| C8 summed request-verifier rate (steps/s) | 249.93 | 363.42 | +45.41% |
| Sieve, five-run median (tok/s) | 356.67 | 375.58 | +5.30% |
| KV pool tokens | 1,546,785 | 4,569,816 | 2.95× |

Sieve min/max: R37 **321.33–367.52**;
R38 **357.33–412.95 tok/s**.
C1 output ranges: R37 213.57–239.17;
R38 256.98–261.55 tok/s.
C8 output ranges: R37 585.62–599.65;
R38 801.31–819.90 tok/s.

Prefill is unchanged within measurement variation (+0.41%). Decode gains
describe the combined source/default changes, not isolated #748 causality.
Boots are sequential and sampling is stochastic; these bounded observations
are not a universal throughput guarantee. KV capacity varies with startup
memory accounting and workload settings.

### Method and correctness boundary

Prefill is six measured 32768-token requests, split equally before/after decode,
with one excluded warmup before each group. Every prompt computes 32768 tokens
locally with zero GPU/external cache hits. Throughput is 32768/TTFT.
Fixed IDs comprise eight nonce bytes mapped to 1000+byte, then 1400+i%127 to
length 32768; unique cache salts prevent reuse. This cyclic input is not a
general natural-text SSD I/O benchmark.

C1/C8 at context 0 use two 30-second measured sweeps. The pinned
[llm-inference-bench client](https://github.com/local-inference-lab/llm-inference-bench/blob/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py)
receives `--decode-warmup-seconds 15`; its readiness-based cell warmups
completed in approximately 5.53 seconds.
The client also performs its pre-decode warmup. No cells report errors,
loops, underfill or warmup timeout. Context 0 means no added context, not an
empty chat template. EOS is respected.

To run the same C1/C8 client profile, first allow at least eight concurrent
requests (qualification used `MAX_NUM_SEQS=32` when starting Compose):

```bash
curl -fL https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py -o llm_decode_bench.py
uv run --with httpx --with rich llm_decode_bench.py \
  --host SERVER --port 8000 --model DeepSeek-V4.1-Flash \
  --display-mode plain --no-hw-monitor --no-resume --respect-eos \
  --token-targeting exact --skip-prefill --concurrency 1,8 --contexts 0 \
  --duration 30 --max-tokens 8192 --temperature 1 \
  --decode-warmup-seconds 15 --output decode-c1-c8.json
```

Decline the client's optional self-update to retain the pinned measurement
code. The server defaults provide top-p .95 and reasoning budget 75.

C8 verifier rate sums per-request verification steps; it is not the physical
engine iteration rate. R38 counts actually verified draft positions where
R37 counted proposals. Therefore raw draft-token acceptance ratios across
these artifacts have different denominators.

Sieve uses `Write a Python script that implements the Sieve of Eratosthenes.`,
temperature 1/top-p .95, budget 75, one excluded warmup and five requests capped
at 2000 output tokens. All measured responses finish with `stop`.
Tokens/s includes reasoning and answer after TTFT. Programs are not executed;
this is not a coding-correctness evaluation.

**Qualified:** arithmetic, pigeon-image recognition and four deterministic
cold/repeated/answer-changing prefix-continuation checks pass in both images.
The R38 components pass 123 selected tests: 16 adaptive, 18 config, 4 allocator,
13 Engram/vLLM, 56 native attention/compressor, 16 disk/B12X. Recipe tests: 211.
Three known bounded-CED oracle cases and the distributed adaptive-publication
test are outside this selection. Test-only `tblib==3.1.0` was installed in a
disposable test container; serving/runtime sources were not modified.

Disk component checks include exact row/scale parity, graph replay, bounded
prefetch and failure cleanup. Whole-model SSD throughput, long-context cache
churn, model quality and the GLM/Qwen/DS4/LMCache serving matrix are not
requalified here. Those entrypoints and retained components remain available.

## Release changes: R37 to R38

- Use merged public JJ and B12X sources, including compatible padded-graph
  pricing for adaptive DSpark verification ([vLLM #748](https://github.com/local-inference-lab/vllm/pull/748)).
- Use configurable sliding-window pages and default main/SWA geometry 256/128
  ([#747](https://github.com/local-inference-lab/vllm/pull/747),
  [#749](https://github.com/local-inference-lab/vllm/pull/749)).
- Preserve decode-row capacity in B12X planning and expose optional NVMe
  Engram scale residency, bounded prefetch and projection partitioning
  ([#736](https://github.com/local-inference-lab/vllm/pull/736),
  [B12X #360](https://github.com/local-inference-lab/b12x/pull/360)).
- Correct verified-versus-proposed speculative-token metrics. Acceptance
  ratios with different denominators must not be compared directly.
- Preserve DS4.1 OMP8, capture128 and one prefill lane while forwarding native
  launcher controls. Keep batch4096, top-p .95 and prefill capture off.
- Rebuild the stable vLLM extension against its committed CMake inputs;
  preserve CUDA13.3/PyTorch2.13, FlashInfer/FlashKDA, LMCache, the three Python
  dependency patches, common model entrypoints and the two-layer layout.

## Source provenance

The [source lock](deepseek-v4.1-flash/r38/source.lock) records full commits,
trees and dependency input/output hashes. The
[registry receipt](deepseek-v4.1-flash/r38/registry.json) verifies the published
digest against the tested image ID and both filesystem layers.

| Component | Source |
|---|---|
| vLLM | Public `dev/jovian-judgement`, `66c293578412417476f842c1da5805d3a3d959a8` |
| B12X | Public `master`, `ce419b52681b7922bb0972d4b58b590a3fd005b2` |
| LMCache | `29bc5a2efde737c436b04499eb62cd1776cebeec`, unchanged |
| Recipe | [Docker #36](https://github.com/local-inference-lab/blackwell-llm-docker/pull/36) |
| Release checklist | [vLLM #745](https://github.com/local-inference-lab/vllm/issues/745) |

No additional unmerged vLLM/B12X source patches are applied.
Model qualification uses revision
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`; the launcher accepts an operator's
staged checkpoint without forcing that revision.

Three hash-locked Python dependency corrections are required in addition to
JJ/B12X: operator-schema enumeration (yingru's
[PyTorch #195110](https://github.com/pytorch/pytorch/pull/195110)), immutable
mutation metadata (equivalent hot-path elimination in Jason Ansel's
[PyTorch #186175](https://github.com/pytorch/pytorch/pull/186175), present in
main at `aea557660`), and CuTe sentinel identity
([CUTLASS #3634](https://github.com/NVIDIA/cutlass/pull/3634)).
The recipe retains its frozen patches; a full dependency upgrade is not
qualified by their upstream status.

Historical results, including explicitly labelled +6000 VRAM measurements:
[R37 release](deepseek-v4.1-flash/r37/release.md),
[R36 release](deepseek-v4.1-flash/r36/release.md).
