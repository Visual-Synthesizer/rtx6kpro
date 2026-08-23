# DeepSeek-V4-Flash-0731 Infernal Invocation r19 Preview

**Status: research-only preview.** This page specifies target-only TP2/DCP1
serving for `deepseek-ai/DeepSeek-V4-Flash-0731`. Infernal Invocation r19 adds
shape-calibrated B12X communication and a TP2 routed-expert launch policy. The
default scheduler profile exposes 2,110,804 effective aggregate KV-cache
tokens while retaining a 4,096-token prefill chunk.

## TL;DR

Download the committed Compose profile, pull the prebuilt image from Docker
Hub, and start the server:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-infernal-invocation-cu133-r19.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r19.yml pull
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r19.yml up -d
```

The Compose profile contains an `image` reference and no `build` section. It
never compiles the runtime locally.

The defaults use two GPUs, target-only decode, B12X W4A8 plus DGLIN, FP8
compressed MLA KV, InstantTensor `BUFFERED`, `MAX_MODEL_LEN=1048576`,
`MAX_NUM_BATCHED_TOKENS=4096`, and GPU memory utilization 0.975. vLLM reports
2,110,804 effective aggregate KV tokens on two 96 GiB RTX PRO 6000 Blackwell
GPUs. The KV cache is resident on the GPUs with `KV_CACHE_DTYPE=fp8_ds_mla`;
LMCache and native vLLM KV offload are disabled by default.

The 2,110,804-token value describes group-aware aggregate capacity. One
request remains limited to 1,048,576 tokens. A single 1,048,002-token prefill
completed, but two concurrent one-million-token requests have not been
qualified.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm174c789-b12x12c4263-fi1ac6942-cu133-torch213-20260823-r19` |
| Registry digest | `sha256:867656d627acaa0f12e7f9069a7de6664805a2cc95d86faef4a7cd288bb84e9d` |
| Image ID | `sha256:85b99258790477201987089984af4d4618f6f342295d3216dc4b307b8911140a` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base revision | `6dc2f516688fe6f84c6994dcd20fddf296853a6c` |
| vLLM integration tree | `174c789e09984049d0d53b261024460ca5e9c449` |
| B12X base revision | `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` |
| B12X integration tree | `12c426322cc5d239023b57a4bd5ab0e60c4302e0` |
| LMCache integration tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2 |
| Compose profile | [`docker-compose-ds4-infernal-invocation-cu133-r19.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-ds4-infernal-invocation-cu133-r19.yml) |

The image contains the exact source under test and requires no source mount.
The topology-calibration changes are research-only and have no public vLLM or
B12X pull request. Pull requests will be prepared only after community
qualification confirms stability on additional PCIe topologies.

## r18 And r19 Behavior

| Area | Infernal Invocation r18 | Infernal Invocation r19 preview |
|---|---|---|
| Target-only all-reduce | Static backend policy | Calibrates B12X and NCCL independently for each reachable row count |
| TP2 B12X transport | Generic one-shot path | Exact peer-push BF16 path for the `[rows, 4096]` model contract |
| TP2 routed experts | Generic launch policy | Shape-qualified persistent-grid policy for 25 through 192 routed rows |
| Default model limit | 524,288 tokens in the general profile | 1,048,576 tokens |
| Default prefill chunk | 8,192 tokens | 4,096 tokens |
| Default serving mode | Fixed probabilistic DSpark K5 | Target-only decode |
| Reported TP2 aggregate KV capacity | Release-profile dependent | 2,110,804 effective tokens under the default profile |

The transport probe measures the exact BF16 `[rows, 4096]` operation used by
the model. It performs stabilization replays, validates output against NCCL,
and requires both a one-percent and 0.25-microsecond median advantage before
selecting B12X. Its cache key includes ordered GPU identity and PCI location,
driver, NCCL binary, B12X source, world size, datatype, hidden size, and row
count. A fingerprint mismatch triggers calibration instead of reusing results
from a different system.

The measured policies were:

| Host | TP | B12X rows | NCCL rows in the measured decode range |
|---|---:|---|---|
| PCIe-switch workstation | 2 | `1-128` | none |
| PCIe-switch workstation | 4 | `1,2,4` | `8,16,24,32` |
| Direct-root-port server | 2 | `1-128` | none |
| Direct-root-port server | 4 | `1,2,4,8,16,24` | `32` |

Unsupported shapes use NCCL. The resulting policy is therefore not a static
assumption that B12X, NCCL, or a PCIe topology wins for every batch shape.

## r18 Comparison

The following TP2 comparison used the same PCIe-switch workstation,
target-only decode, DCP1, B12X W4A8 plus DGLIN, FP8 compressed MLA KV,
`MAX_NUM_SEQS=32`, graph cap 128, and
`MAX_NUM_BATCHED_TOKENS=4096`.

| Image | C1 | C2 | C4 | C8 | C16 | C32 | 8K prefill | 64K prefill | 128K prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infernal Invocation r18 | 146.4 | 238.6 | 389.6 | 566.0 | 801.7 | 1,224.5 | 10,901 | 13,351 | 12,304 |
| Infernal Invocation r19 preview | 150.4 | 245.4 | 396.1 | 582.7 | 875.8 | 1,309.2 | 13,189 | 13,584 | 12,280 |
| r19 relative to r18 | +2.7% | +2.8% | +1.7% | +3.0% | +9.2% | +6.9% | +21.0% | +1.7% | -0.2% |

The measured gain is not limited to C1. Decode improved at every tested
concurrency. The 128K prefill difference is within measurement noise.

## Lucifer Comparison

The comparable Lucifer control used the same host, scheduler envelope, model,
target-only mode, and calibrated B12X all-reduce. This isolates model-compute
dispatch rather than assigning the collective difference to Lucifer.

| Backend | C1 | C2 | C4 | C8 | C16 | C32 | 8K prefill | 64K prefill | 128K prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infernal Invocation r19 B12X | 150.4 | 245.4 | 396.1 | 582.7 | 875.8 | 1,309.2 | 13,189 | 13,584 | 12,280 |
| Lucifer compute | 132.5 | 211.1 | 351.8 | 575.4 | 851.6 | 1,237.6 | 12,887 | 9,418 | 7,709 |
| B12X relative to Lucifer | +13.5% | +16.2% | +12.6% | +1.3% | +2.8% | +5.8% | +2.3% | +44.2% | +59.3% |

Infernal Invocation r19 B12X is faster in every cell of the comparable
same-host matrix. It also exceeds every recorded historical target-only
Lucifer decode result from C1 through C32.

Historical Lucifer measurements used several scheduler envelopes. Their
per-column maxima are shown separately so the comparison does not imply that
all values came from one launch:

| Profile | C1 | C2 | C4 | C8 | C16 | C32 | 8K prefill | 64K prefill | 128K prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical Lucifer maximum | 132.5 | 211.1 | 352.0 | 575.4 | 856.9 | 1,267.4 | 13,508 | 12,788 | 11,705 |
| r19 capacity profile, 4,096-token chunks | 150.4 | 245.4 | 396.1 | 582.7 | 875.8 | 1,309.2 | 13,189 | 13,584 | 12,280 |
| r19 prefill-throughput profile, 8,192-token chunks | 150.4 | 245.2 | 397.9 | 583.7 | 881.2 | 1,315.1 | 14,741 | 13,865 | 12,781 |

The r19 capacity profile is 2.4 percent below the historical Lucifer maximum
at 8K prefill while exceeding the historical 64K and 128K maxima. The optional
8,192-token profile exceeds all historical Lucifer maxima in this table but
provides less resident KV capacity.

## Platform Measurements

The platform matrix used target-only decode, DCP1, B12X W4A8 plus DGLIN, FP8
compressed MLA KV, InstantTensor `BUFFERED`, `MAX_NUM_SEQS=32`, graph cap 128,
`MAX_MODEL_LEN=262144`, `MAX_NUM_BATCHED_TOKENS=8192`, and GPU memory
utilization 0.95. Decode cells ran for 30 seconds after a five-second warmup.
Prefill cells used exact token inputs and 15 cold samples.

Aggregate output throughput in tokens per second:

| Host | TP | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCIe-switch workstation | 2 | 150.4 | 245.2 | 397.9 | 583.7 | 881.2 | 1,315.1 |
| PCIe-switch workstation | 4 | 182.6 | 309.1 | 513.5 | 736.3 | 1,166.4 | 1,791.8 |
| Direct-root-port server | 2 | 130.4 | 210.9 | 347.0 | 515.3 | 782.4 | 1,182.4 |
| Direct-root-port server | 4 | 149.8 | 258.8 | 437.6 | 670.0 | 1,027.3 | 1,593.6 |

Cold input throughput in tokens per second:

| Host | TP | 8K | 64K | 128K |
|---|---:|---:|---:|---:|
| PCIe-switch workstation | 2 | 14,741 | 13,865 | 12,781 |
| PCIe-switch workstation | 4 | 16,865 | 15,941 | 14,657 |
| Direct-root-port server | 2 | 13,524 | 13,483 | 12,354 |
| Direct-root-port server | 4 | 16,263 | 15,408 | 14,119 |

The hosts use different RTX PRO 6000 Blackwell variants as well as different
PCIe topologies, so host-to-host deltas are not a pure switch-versus-root-port
measurement.

## KV Capacity And Prefill

`MAX_NUM_BATCHED_TOKENS` controls both scheduler work per step and transient
memory admission. The asynchronous scheduler can retain two batches in flight,
so changing the value affects activation storage and bounded cache groups in
addition to changing the prefill chunk.

The following TP2 profiles used `MAX_MODEL_LEN=1048576` and GPU memory
utilization 0.975:

| Batched-token limit | Reported effective KV tokens | C1 | 8K prefill | 256K prefill | 1,048,002-token prefill |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,227,878 | 147.9 | 14,590 | 10,936 | 5,723 |
| 4,096, r19 default | 2,110,804 | 147.7 | 12,411 | 10,755 | 5,688 |
| 2,048 | 2,835,105 | 147.5 | not measured | 7,094 | 5,134 |

The 4,096-token default preserves C1 and 256K-to-1M prefill throughput within
1.7 percent of the 8,192-token profile while increasing reported aggregate KV
capacity by 71.9 percent. The 8K prefill tradeoff is 14.9 percent. A 2,048-token
chunk recovers more capacity but materially reduces long-prefill throughput,
so it is not the default.

Use the throughput profile when short-prefill latency matters more than
resident capacity:

```bash
MAX_NUM_BATCHED_TOKENS=8192 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r19.yml up -d
```

The decline near one million cached tokens is dominated by sparse-indexer
score scanning and top-k selection, not by the all-reduce backend. A matched
NCCL control was slower than B12X at C1, 8K, 256K, and 1,048,002-token
prefill. Detailed CUDA profile evidence is documented in
[B12X PCIe transport calibration](ds4f-b12x-pcie-autotune.md).

## Validation

| Gate | Result |
|---|---|
| B12X focused unit suite | 276 passed, 1 skipped |
| vLLM helper and custom-all-reduce suite | 34 passed, 1 skipped |
| Docker release tests | r18 compatibility and r19 source-lock tests passed |
| Source reconstruction | vLLM and B12X patches reproduced the locked integration trees exactly |
| All-reduce correctness | Every calibration row matched the NCCL reference |
| End-to-end matrix | TP2 and TP4 decode and prefill completed on switch and direct-root hosts |
| Published-image smoke | TP2 startup, CUDA graph capture, and C1 completed at 147.7 tok/s; vLLM reported 2,110,804 effective KV tokens |
| TP2 long context | Exact 1,048,002-token prefills completed with 2,048-, 4,096-, and 8,192-token chunks |

## Qualification Limits

- **Implemented:** exact TP2 peer-push BF16 all-reduce, per-shape B12X/NCCL
  calibration, TP2 routed-expert launch selection, and the TP2 capacity
  profile.
- **Research-only:** unattended deployment and the source changes identified
  by the integration-tree hashes.
- **Unsupported by this evidence:** DSpark K5 or K7 performance, MTP, DCP
  greater than one, TP8, TP16, two concurrent one-million-token requests, and
  model-quality claims.
- Preserve the release-scoped `/cache` volume. The calibration cache is bound
  to the ordered GPU and software fingerprint, and B12X compiles uncovered
  shapes on first use.
- Native vLLM KV offload and LMCache are disabled by the Compose defaults.
  Both remain available as explicit deployment choices but were not part of
  the 2,110,804-token resident-capacity measurement.
