# Qwen3.8-27B on RTX PRO 6000 Blackwell

This page records Qwen3.8-27B throughput reported on NVIDIA RTX PRO 6000
Blackwell systems. Results are grouped by tensor-parallel (TP) size so that a
single-GPU result is not presented as directly comparable to a two-GPU or
four-GPU deployment.

**Evidence snapshot:** 2026-08-21

**Status:** `research-only` except for the explicitly marked TP4 official-FP8
configuration, which is `qualified` against repeated performance and quality
gates.

The research-only dense-MLP QSRT K5 checkpoint, its weighted-SVD rank-16
recovery, all-boundary optimizer evaluation, packed B12X runtime, and exact
source inventory are documented in
[Qwen3.8-27B QSRT K5 dense-MLP recovery](qwen38-qsrt-k5-r16.md).

## Reading The Tables

- `Prefill` is prompt-processing throughput in input tokens per second.
- `C1`, `C2`, and `C4` are aggregate output-token throughput at concurrency 1,
  2, and 4.
- `ctx` is the number of existing context tokens before decode begins.
- Multi-token prediction (MTP) changes decode throughput, acceptance, physical
  batch sizes, and the required CUDA graph ladder. MTP-on and MTP-off rows are
  not interchangeable.
- FP8 is 8-bit floating point, BF16 is bfloat16, NVIDIA FP4 (NVFP4) is the
  Blackwell 4-bit floating-point format, and W8A8 means 8-bit weights with
  8-bit activations.
- KV means key/value cache, TTFT means time to first token, KLD means
  Kullback-Leibler divergence, and PCIe means Peripheral Component Interconnect
  Express.
- A dash means that the source did not publish a result for that cell.
- Compare values only when checkpoint, runtime, speculation, context,
  concurrency, power limit, and PCIe topology are sufficiently similar.

The repository name `lribeiro/Qwen3.8-27B-nvfp4-v17` retains an `nvfp4`
label, but its model card defines a uniform W8A8 FP8 GPTQ checkpoint, not an
NVFP4 checkpoint.

## Configuration Summary

| TP setting | Checkpoint and runtime | Speculation | Prefill sample | C1 sample | C4 sample | Status |
|---:|---|---|---:|---:|---:|---|
| `TP=1` | Official FP8, vLLM Gilded Gnosis r31 | MTP3 | 7,686 at 8K | 77.8 at ctx0 | 292.7 at ctx0 | `research-only` |
| `TP=1` | EXL3 K5/K6, vLLM, PCIe 5.0 x8 Max-Q | MTP3 | 3,466 at 8K | 78.0 at ctx0 | - | `research-only` |
| `TP=1` | W8A8 FP8 GPTQ v17, vLLM 0.26.1rc0 | off | 7,306 at 2K | 49.8 at ctx0 | 180.6 at ctx0 | `research-only` |
| `TP=2` | W8A8 FP8 GPTQ v17, vLLM | MTP3 | 11,851 at 8K | 102.5 at ctx0 | 393.5 at ctx0 | `research-only` |
| `TP=2` | Official FP8, tuned vLLM/FlashInfer | MTP3 | 7,200 at 8K | 120.8 at ctx0 | - | `research-only` |
| `TP=2` | Official FP8, SGLang/EAGLE | EAGLE | 7,325 at 8K | 141.0 at ctx8K | 507.4 at ctx8K | `research-only` |
| `TP=4` | Official FP8, vLLM, FP8 KV, full graph ladder | MTP3 | - | 135.8 sustained | 464.6 sustained | `qualified` |

This summary is a navigation aid, not a ranking. Detailed conditions and
context curves follow.

## TP=1: One GPU

### Official FP8 With MTP3, TP=1

The following result used `Qwen/Qwen3.8-27B-FP8` on one 600 W RTX PRO 6000
workstation GPU. The server used vLLM Gilded Gnosis r31, MTP3, FP8 KV cache,
InstantTensor loading, 48 GiB native KV offload, `max_num_seqs=4`, and prefix
caching. Tensor parallelism was `TP=1`, passed to vLLM as
`--tensor-parallel-size 1`.

**Status:** `research-only`; the source is one community benchmark capture
without repeat variance or quality-gate receipts.

| Metric | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Prefill tok/s | 7,686 | 7,562 | 6,902 | 5,877 | 4,439 |

| Decode | ctx0 | ctx16K | ctx32K | ctx64K | ctx128K |
|---|---:|---:|---:|---:|---:|
| C1 tok/s | 77.8 | 76.2 | 73.9 | 81.7 | 69.8 |
| C2 tok/s | 149.2 | 149.6 | 149.3 | 138.7 | 145.9 |
| C4 tok/s | 292.7 | 296.5 | 279.7 | 264.7 | - |

Source: [Discord benchmark and complete launch command](https://discord.com/channels/1466898002793857221/1528331644933767190/1537926814008213636).

### W8A8 FP8 GPTQ V17, TP=1

The `lribeiro/Qwen3.8-27B-nvfp4-v17` checkpoint uses FP8 E4M3 weights and
dynamic FP8 activations. The vision tower and MTP head remain BF16. The model
card reports one RTX PRO 6000, vLLM 0.26.1rc0, FlashInfer, temperature 0, and
`ignore_eos` for the MTP-off duration test. Effective tensor parallelism was
`TP=1`, equivalent to vLLM `--tensor-parallel-size 1`.

**Status:** `research-only`; the model-card result does not include the same
long-context matrix as the official-FP8 capture.

| Metric | ctx0 | ctx2K |
|---|---:|---:|
| Prefill tok/s | - | 7,306 |
| TTFT | - | 0.279 s |
| C1 tok/s | 49.8 | 49.4 |
| C4 tok/s | 180.6 | 176.1 |

The checkpoint configuration must ignore `mtp.*` during runtime activation
quantization. With that contract in place, the model card reports 70-77%
acceptance and a separate C1 comparison of 106.9 tok/s with MTP versus 69.4
tok/s for its comparison arm. That MTP result uses a different measurement
from the MTP-off table and is not merged into the context matrix.

Source: [W8A8 FP8 GPTQ v17 model card](https://huggingface.co/lribeiro/Qwen3.8-27B-nvfp4-v17).

### EXL3 K5/K6 Versus NVFP4, TP=1

This same-host comparison used one RTX PRO 6000 Max-Q connected at PCIe 5.0
x8 and MTP3. It is useful because both checkpoints share the host and harness;
it does not establish performance on a full-width PCIe link. Both arms used
`TP=1`, equivalent to vLLM `--tensor-parallel-size 1`.

**Status:** `research-only`.

| Metric | EXL3 K5/K6 | NVFP4 with FlashInfer |
|---|---:|---:|
| Prefill 8K tok/s | 3,466 | 8,988 |
| Prefill 32K tok/s | 3,256 | 7,094 |
| Prefill 128K tok/s | 2,497 | 4,014 |
| C1 ctx0 tok/s | 78.0 | 90.5 |
| C8 ctx0 tok/s | 373.2 | 672.3 |
| C8 ctx16K tok/s | 237.6 | 666.3 |

Source: [Discord same-host EXL3/NVFP4 capture](https://discord.com/channels/1466898002793857221/1528331644933767190/1538168616166367263).

## TP=2: Two GPUs

### W8A8 FP8 GPTQ V17 With MTP3, TP=2

The following matrix used two RTX PRO 6000 GPUs, MTP3, and a reported GPU
memory utilization of 0.40. Tensor parallelism was `TP=2`, passed to vLLM as
`--tensor-parallel-size 2`.

**Status:** `research-only`; exact card power, PCIe topology, client duration,
and repeat variance were not published with the capture.

| Metric | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Prefill tok/s | 11,851 | 11,538 | 10,683 | 9,293 | 7,372 |

| Decode | ctx0 | ctx16K | ctx32K | ctx64K | ctx128K |
|---|---:|---:|---:|---:|---:|
| C1 tok/s | 102.5 | 108.8 | 103.8 | 103.9 | 103.4 |
| C2 tok/s | 193.9 | 181.8 | 191.4 | 187.0 | 175.2 |
| C4 tok/s | 393.5 | 391.4 | 368.5 | 357.7 | 322.9 |
| C8 tok/s | 690.7 | 676.3 | 650.6 | 595.4 | - |
| C16 tok/s | 1,199.3 | 1,102.0 | 1,023.0 | - | - |
| C32 tok/s | 1,824.6 | 1,635.5 | - | - | - |

Source: [Discord TP2 W8A8 FP8 GPTQ capture](https://discord.com/channels/1466898002793857221/1538962909672112208/1538977882896994314).

### Official FP8: Attention Backend Comparison, TP=2

These results used two 350 W RTX PRO 6000 workstation GPUs, official FP8
weights, and MTP3. They show why the effective attention backend and runtime
build must accompany every throughput claim. Every row used `TP=2`, passed to
vLLM as `--tensor-parallel-size 2`.

**Status:** `research-only`.

| Runtime path | Prefill 8K | Prefill 64K | Prefill 128K | C1 ctx0 | C1 ctx64K | C1 ctx128K |
|---|---:|---:|---:|---:|---:|---:|
| vLLM recipe without explicit FlashInfer | 7,639 | 6,137 | 5,045 | 126.7 | 52.2 | 31.9 |
| vLLM recipe with `--attention-backend flashinfer` | 7,621 | 5,983 | 4,863 | 114.0 | 100.8 | 87.5 |
| Compiled Gilded Gnosis vLLM with FlashInfer | 7,200 | 5,992 | 4,777 | 120.8 | 116.7 | 109.4 |

The row without explicit FlashInfer has the highest ctx0 value but loses most
of its decode rate as context grows. It is therefore not the preferred
long-context path.

Sources: [implicit backend](https://discord.com/channels/1466898002793857221/1528331644933767190/1537889649249493087), [explicit FlashInfer](https://discord.com/channels/1466898002793857221/1528331644933767190/1537901135489404999), [compiled Gilded Gnosis runtime](https://discord.com/channels/1466898002793857221/1528331644933767190/1537915668727332924).

### Official FP8 With SGLang/EAGLE, TP=2

This external-runtime reference used SGLang with EAGLE on two RTX PRO 6000
Max-Q GPUs. It is included to define the reported cross-runtime envelope; it
does not isolate engine effects from graph, speculation, power, or launch
configuration. Tensor parallelism was `TP=2`, passed to SGLang as
`--tp-size 2`.

**Status:** `research-only`.

| Metric | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Prefill tok/s | 7,325 | 7,027 | 6,710 | 6,091 | 5,151 |

| Decode | ctx8K | ctx16K | ctx32K | ctx64K | ctx128K |
|---|---:|---:|---:|---:|---:|
| C1 tok/s | 141.0 | 137.6 | 133.0 | 125.4 | 115.4 |
| C4 tok/s | 507.4 | 536.1 | 482.2 | 438.2 | 368.0 |
| C8 tok/s | 893.7 | 884.6 | 812.5 | 715.6 | 547.1 |
| C16 tok/s | 1,451.4 | 1,303.9 | 1,209.8 | 944.0 | - |
| C32 tok/s | 1,869.1 | 1,727.0 | 1,476.7 | - | - |

Source: [Discord SGLang/EAGLE TP2 capture](https://discord.com/channels/1466898002793857221/1528331644933767190/1537934075573043401).

## TP=4: Four GPUs

### Qualified Official-FP8 Deployment, TP=4

The qualified configuration uses four RTX PRO 6000 Blackwell 96 GB GPUs,
official FP8 weights, FP8 KV cache, static YaRN for a 1M-token context window,
MTP3, and a full MTP-aligned decode CUDA graph ladder from 4 through 256
physical rows. The multimodal processor uses `min_pixels=131072`. Tensor
parallelism was `TP=4`, passed to vLLM as `--tensor-parallel-size 4`.

**Status:** `qualified`.

| Configuration | C1 tok/s | C2 tok/s | C4 tok/s | Geometric-mean change |
|---|---:|---:|---:|---:|
| BF16 TP4 baseline | 110.7 | 215.3 | 402.0 | baseline |
| Official FP8, FP8 KV, MTP3 | 135.8 | 248.1 | 464.6 | +17.77% |
| Relative change | +22.6% | +15.2% | +15.6% | +17.77% |

Qualification conditions:

- two performance repeats with coefficient of variation at or below 1.65%;
- exact-token retrieval at 8K, 256K, 512K, and 960K;
- sustained decode checks at 300K, 512K, and 960K;
- tool-call and 40-case task-retention gates;
- vision score 24/30;
- full-vocabulary KLD fidelity checks.

The report rejects a single-size MTP4 CUDA graph because longer concurrent
runs fell back outside the captured physical batch. It also rejects LMCache
for this recurrent/full-attention hybrid profile because the required
`max_num_batched_tokens=1600` reduced prefill throughput by approximately 50%.

Source: [TP4 qualification report](https://discord.com/channels/1466898002793857221/1528331644933767190/1539032445515726970).

### Coding Workload Reference, TP=4

A separate BF16 TP4 coding request emitted 772 completion tokens at 172.58
tok/s including time to first token and 174.61 tok/s for generation only.
Because this is a workload trace rather than a fixed C1 duration benchmark, it
must not replace the qualified C1 row above. Tensor parallelism was `TP=4`,
equivalent to vLLM `--tensor-parallel-size 4`.

**Status:** `research-only`.

Source: [Discord BF16 TP4 coding trace](https://discord.com/channels/1466898002793857221/1528331644933767190/1537884251029114961).

## MTP-Off Quantization Sweep

A common MTP-off harness measured 40 Qwen3.8-27B checkpoints for KLD, top-1
agreement, prefill, and decode. The source does not identify GPU count,
topology, context shape, or runtime revision. The absolute throughput values
therefore cannot be assigned to TP1, TP2, or TP4 and cannot be compared with
the topology-specific tables above.

**Status:** `research-only`.

Representative rows from the sweep:

| Checkpoint | Format | KLD | Top-1 | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|
| Local `nvfp4-gptq-v14` | NVFP4 GPTQ | 0.015896 | 0.9567 | 7,742 | 52.8 |
| Local `nvfp4-gptq-v6` | NVFP4 GPTQ | 0.015972 | 0.9564 | 7,756 | 52.3 |
| Local `nvfp4-gptq-v7` | NVFP4 GPTQ | 0.019752 | 0.9516 | 8,020 | 54.1 |
| Local `nvfp4-gptq-v4` | NVFP4 GPTQ | 0.025964 | 0.9448 | 8,488 | 56.5 |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4 | 0.094946 | 0.9054 | 9,338 | 62.6 |
| `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` | mixed INT4 | 0.060887 | 0.9230 | 4,615 | 70.5 |
| Official/local FP8 baseline | FP8 | 0.013348 | 0.9615 | 2,953 | 23.8 |
| `Vishva007/Qwen3.8-27B-W4A16-AutoRound` | W4A16 | 0.073144 | 0.9154 | 4,632 | 74.9 |
| `malaiwah/Qwen3.8-27B-EXL3-K5K6` | EXL3 K5/K6 | 0.008170 | 0.9697 | 1,679 | 22.3 |
| `shawnw3i/Qwen3.8-27B-AWQ-MTP` | AWQ | 0.080480 | 0.9113 | 4,564 | 76.4 |
| `devan-carlin/Qwen3.8-27B-int4-AutoRound` | INT4 | 0.084265 | 0.9088 | 4,723 | 76.6 |
| `malaiwah/Qwen3.8-27B-K4` | EXL3 K4 | 0.030696 | 0.9452 | 1,468 | 24.3 |

Source: [complete 40-checkpoint Discord table](https://discord.com/channels/1466898002793857221/1528331644933767190/1538616626629447721).

## Unsupported Comparisons

- An NVFP4 plus DSpark result near 283 tok/s entered a repetition loop. It is
  an invalid correctness run and is not a performance result.
- A high-concurrency EXL3/NVFP4 comparison reported C8 through C64 but omitted
  TP size, context, hardware, and runtime. It is not assigned to a TP section.
- The `shisa-ai/Qwen3.8-27B-FP8-BLOCK` discussion contains KLD evidence but no
  throughput matrix.
- Results from RTX 5090 systems are not used as RTX PRO 6000 headline values.

## Publication Contract

A result can replace a `research-only` row with `qualified` status when its
artifact identifies:

1. checkpoint repository and immutable revision;
2. Docker image or runtime source revisions;
3. GPU model, power limit, PCIe topology, and TP size;
4. KV-cache format, MTP/EAGLE settings, graph sizes, and attention backend;
5. prompt length, decode context, concurrency, output length, and sampling;
6. warmup policy, repeated measurements, and variance;
7. correctness checks appropriate to text, tools, vision, and long context.
