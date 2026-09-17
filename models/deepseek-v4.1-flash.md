# DeepSeek-V4.1-Flash

Serve `deepseek-ai/DeepSeek-V4.1-Flash` using the `ds41-flash` profile in the
[shared Docker guide](../docs/unified-vllm-docker.md). It supports native
text/vision input, B12X kernels and the checkpoint's embedded DSpark draft.
It is distinct from [DeepSeek V4 text](deepseek-v4-flash.md) and
[V4 Vision](deepseek-v4-flash-vision.md).

Status: **implemented** profile, with **qualified** bounded TP4/DCP1,
RAM-Engram measurements and API/image tests below. Whole-model disk throughput,
DCP4 and external KV offload are not qualified by these checks.

## Start the server

Select `LIL_IMAGE` from the [shared image section](../docs/unified-vllm-docker.md#select-the-image).
For the RAM-Engram placement used in the measurements, choose:

```bash
PROFILE=ds41-flash
GPU_DEVICES=0,1,2,3
TP=4
PORT=8000
SERVE_ARGS=(--mode dspark --draft-tokens 7 --engram-table-memory ram)
```

Run the [common launch command](../docs/unified-vllm-docker.md#start-a-server).
It adds the DS4.1 memlock/io_uring permissions and downloads the checkpoint
into the shared Hugging Face cache. The API name is `DeepSeek-V4.1-Flash`.

For the profile's disk-backed table default, use:

```bash
SERVE_ARGS=(--mode dspark --draft-tokens 7 --engram-table-memory disk)
```

For target-only serving with RAM tables:

```bash
SERVE_ARGS=(--mode off --engram-table-memory ram)
```

These are alternatives, not arguments to combine. DS4.1's profile supports
`off` and `dspark`, not GLM's MTP/DFlash2 modes. The default maximum concurrency
is four; append `--max-num-seqs 32` to reproduce the C8 benchmark admission.

## RAM versus SSD ngrams

Engram contains learned n-gram model tables. It is **not** KV cache, prefix
caching or an LMCache tier. Target/draft transformer weights remain on GPU.

| Placement | Behavior | Capacity requirement |
|---|---|---|
| `disk` | Native io_uring reads selected checkpoint rows from local SSD into bounded staging buffers | Fast local SSD/NVMe; loader, OS cache and staging still need RAM |
| `ram` | Complete tables in pinned, GPU-mapped host RAM | Historical TP4 checkpoint accounting: 188.83 GiB for tables, plus loader/server/OS memory |

RAM allocation failure does not silently choose disk. Generic CPU model
offload remains disabled. Optional `--engram-disk-resident-scales` and
`--engram-projection-tp` are not enabled by default and have no whole-model
speed claim in this qualification. `CACHE_MODE=lmcache` is unsupported for
this profile; selecting RAM Engram does not enable it.

## Serving defaults

| Setting | Profile behavior |
|---|---|
| Parallelism / speculation | TP4/DCP1, DSpark K7 with adaptive verification |
| Draft sampling | Greedy proposals, standard rejection; target sampling remains temperature 1/top-p .95 |
| Backends | B12X attention, MoE and dense; B12X/NCCL communication |
| Scheduler | 4096 tokens, four sequences, one prefill lane, compute share 0.4 |
| Context / GPU fraction | 131,072 / 0.95 |
| Main / sliding-window pages | 256 / 128 tokens |
| Graphs | Full-and-piecewise decode, cap 128; breakable prefill off |
| Prefix cache | Enabled; retention interval defaults to `0`, not cache disabled |
| Reasoning | `high` by default; publisher mapping `low=50`, `high=75`, `max=100` |
| API compatibility | DS4.1 tool namespaces/reminders and Responses text-part normalization |

The native cache is heterogeneous despite the CLI's `fp8` label: MXFP8
sliding-window payloads, NVFP4 indexed payloads and index/state groups. Do not
describe it as a uniform FP8 cache. The GLM/Qwen request-boundary recurrent
adapter does not apply to DeepSeek's attention-cache structure.

Use `--adaptive-verification=false` to disable DSpark trimming, not DSpark
itself. `--adaptive-verification-cost-scale` changes trimming cost; neither
alternative is timed here. A request can override the reasoning budget with
`"chat_template_kwargs":{"reasoning_effort":50}`.

## Qualification and measurements

Stock RTX PRO 6000 Workstation quartet, TP4/DCP1, RAM Engram, adaptive DSpark
K7, 4096-token budget, 32 sequences, FP8 CLI cache mode, target temperature
1/top-p .95. Context-zero decode is the median of three warmed 30-second
runs. Prefill is uncached nominal 32K measured from client TTFT. C8 is aggregate.

| Metric | Community R38 → wheel image | Change |
|---|---:|---:|
| C1 output | 250.87 → 256.59 tok/s | +2.28% |
| C8 output | 808.17 → 830.82 tok/s | +2.80% |
| 32K prefill | 20,079 → 20,234 tok/s | +0.77% |
| Logical KV tokens | 4,503,190 → 4,714,406 | +4.69% |

[Image/source identities and raw measurements](../benchmarks/prepared-b12x-serving/).
The memory/workspace fixes are included; these are not figures for disk Engram
or a four-concurrent-request configuration. No Sieve rerun is claimed for the
wheel-image comparison; R38 Sieve results remain in the archive.

### Indexed PCIe plan lookup

The bounded lookup change is tested separately with only the communicator
source differing: C1 **247.61 → 254.92 tok/s**, C8 **799.07 → 799.57**, and
prefill **20,471 → 20,577**. A 512-declaration CPU miss falls **139 → 0.70 µs**.
C1 acceptance changes 2.443 → 2.551 while request-verifier throughput falls
1.45%; the emitted-token gain is not an isolated lookup speedup.
[Component evidence](../benchmarks/prepared-b12x-contracts/#declared-plan-index-ds41).

### Breakable prefill

`VLLM_USE_BREAKABLE_CUDAGRAPH=1` captures graph segments around dynamic
attention/cache operations, which still execute outside those segments. It is
not a single FULL graph for all prefill work. Decode graphs remain available
when this option is off.

The same-image flag-only comparison measures **20,577 → 20,801 tok/s
(+1.09%)** prefill, but logical KV falls **4,763,329 → 3,975,153 tokens
(−16.55%)** as graph memory grows. The option remains off by default. The
passing replay checks use caller-owned buffers, not a process-global pool.
[Measured trade-off](../benchmarks/prepared-b12x-contracts/#breakable-prefill-ds41).

## API and image checks

The composed tokenizer/frontend/parser suite passes 187 tests. Fifteen HTTP
cases cover tool namespaces, reminders, history, named choices and malformed
calls; six Responses cases cover strings, typed text, streaming, assistant/tool
history and an image. These are not general model-quality or repetition tests.

Two, eight and sixteen 128×128 images and image-history checks pass without an
artificial image-count override. Repeated history reuses 8192 prefix tokens;
the changed-image case answers correctly but records zero prefix hits. Do not
claim changed-image prefix reuse or arbitrary-resolution capacity from that test.

## Historical releases and source review

- [Community R38 deployment and measurement archive](deepseek-v4.1-flash-community-r38.md):
  R37/R38 measurements, Sieve, source/dependency locks and exact checkpoint scope.
- [R37 record](deepseek-v4.1-flash/r37/release.md) and
  [R36 record](deepseek-v4.1-flash/r36/release.md), including separately labelled clocks.
- [Shared runtime dependency corrections](https://github.com/local-inference-lab/blackwell-llm-docker/blob/b3fe0afe1621273059fb19dee1034e2272043a55/runtime/DEPENDENCIES.md):
  PyTorch/CuTe corrections belong to image assembly, not model startup patches.
- [Issue #773](https://github.com/local-inference-lab/vllm/issues/773):
  open source PRs, integration status and qualification limits.
