# Qwen TP1 engine throughput comparison

Status: **qualified for the measured vLLM and SGLang cells**. Operator-reported
mratsim turbo values are **research-only** and were not independently reproduced.
Measurements were recorded on 2026-09-08. No serving or clock changes are part
of this documentation publication.

## Artifact and deployment contract

The target is `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`, revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed` in both measured deployments.
Each engine uses one 96 GB RTX PRO 6000 Blackwell Workstation GPU, stock memory
and graphics offsets, and a 600 W limit. The 185 recorded offset observations
contain no overclock. GPUs were not swapped; silicon differences are included.

| Setting | vLLM | SGLang FlashInfer |
|---|---|---|
| Physical GPU | 0 | 1 |
| Image ID | `sha256:259a592cc1b86312a9bb61bd95c1aad66577d468a5502b4191ec69e22313fede` | `sha256:07bb7e0f354c606cf77dabc49be95aebb8b2b52a5ba2cc91cc4062c1e93693e3` |
| Speculation | MTP3 | NEXTN, three steps, four draft-tree tokens |
| Target vocabulary head | BF16 | BF16 |
| Draft vocabulary head | Private NVFP4 W4A16 copy | BF16 |
| Attention KV | FP8 | FP8 E4M3 |
| Recurrent state | FP32 | BF16 |
| Host PLE offload | Enabled | Enabled |
| MoE kernels | B12X | FlashInfer CUTLASS |
| Concurrent active requests | 16 | 4 |
| Prefill token budget | 6,019 | 4,096 |
| Maximum request context | 262,144 | 262,144 |
| External KV cache | None | 32 GiB hierarchical cache, write-through |

The comparison does not isolate a single kernel, quantization setting or engine
implementation. The [vLLM Compose recipe](../qwen38-flash-next.compose.yml)
documents the supported launch interface and explicit draft-head precision.

## Decode and prefill

Decode uses llm-inference-bench 0.6.1, context zero with 119 identical rendered
input token IDs, 15 seconds of warmup and one 30-second window per concurrency.
Requests use **temperature 1, top-p 0.95, top-k 20, reasoning xhigh and EOS
termination**. All measured cells have zero errors, no underfill and no
detected exact loops. A concurrency above one reports aggregate output.

| Measurement, tok/s | vLLM | SGLang FlashInfer | mratsim turbo, reported |
|---|---:|---:|---:|
| C1 | 172.8 | 152.5 | 154.4 |
| C2 | 304.3 | 267.5 | Not measured |
| C4 | 485.4 | 446.8 | Not measured |
| C8 | 632.4 | Not measured | Not measured |
| C16 | 944.5 | Not measured | Not measured |
| Uncached 32K prefill | 14,813.4 | 15,582.7 | 15,281 |

Prefill submits the same 32,768 explicit token IDs, requests one output token,
discards two warmups, and reports the median of five requests with zero cache
hits. Input tokens divided by complete HTTP wall time includes first-output
work. The distinct vLLM engine-accounted median is 14,960.7 tok/s.

SGLang's four-active-request cap prevents interpreting queued C8/C16 traffic
as eight or sixteen simultaneous decodes. Those cells remain unmeasured.

## Sieve coding

One discarded warmup precedes ten measured sequential C1 requests per engine.
The prompt is `Write a Python script that implements the Sieve of Eratosthenes.`
Sampling matches decode: **temperature 1**, top-p 0.95, top-k 20, reasoning
xhigh. Each request permits at most 2,000 output tokens; all twenty measured
requests finish by EOS. Rates include reasoning and answer tokens and exclude
time to first text, following the benchmark's Coding Peak timing.

| Engine | Median tok/s | Minimum tok/s | Maximum tok/s |
|---|---:|---:|---:|
| vLLM | 239.8 | 222.1 | 265.1 |
| SGLang FlashInfer | 205.3 | 176.2 | 232.6 |

The [JSON summary](tp1-engine-comparison.json) includes all ten rates and output
lengths per engine. Generated Python was not executed. Neither throughput nor
EOS completion constitutes a model-quality qualification.

## Operator-reported turbo scope

The serving operator supplied mratsim turbo C1/context-zero output of 154.4
tok/s and uncached 32K input throughput of 15,281 tok/s on 2026-09-08. No source
or immutable image identity, matching hardware/clock record, sampling proof,
repetition count, C2+ cells or Sieve measurements accompany these values.
They remain attributed observations, not independent tests or proof of a
statistically significant difference from either measured deployment.
