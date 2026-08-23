# DeepSeek-V4-Flash B12X PCIe Transport Calibration

**Status: research-only.** This page specifies a topology-calibrated B12X
all-reduce and shape-qualified B12X MoE runtime for
`deepseek-ai/DeepSeek-V4-Flash-0731`. The implementation selects B12X or NCCL
for each measured all-reduce row count instead of assigning one transport to
an entire machine. Measurements cover target-only TP2 and TP4 serving on RTX
PRO 6000 Blackwell GPUs.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-r18-pcie-autotune-b12x57b5ba9-20260823-exp2` |
| Registry digest | `sha256:ea4f3f7eb3b818c9a916519ab3fef4ba0b6c85f68afa4321d8c139bfa01fd696` |
| Image ID | `sha256:df712ed842cd5db83be5ece4836505d3646aa7e17ad7fad4d7ce3edeb321a8b0` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base revision | `b5f995e73e6b7fe27c9927477e277a151ebcc9e9` |
| vLLM source digest | `9c61ec00d077c6cee9b847f207ed6de216e06e6eb6dfcf0584a46c8030dbb310` |
| B12X base revision | `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` |
| B12X source digest | `98dcc5dd6d4449e04e4bafc5f5c0899148c1aeed7640ca805aa565fbd077e5a1` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2 |

The image contains the source under test and requires no runtime source mount.
The vLLM and B12X changes are unmerged proof-of-concept source, identified by
the immutable labels above. Do not use this image as a qualified release.

## Start The Server

Use the Infernal Invocation r18 Compose profile and override its image:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-infernal-invocation-cu133-r18.yml

IMAGE=voipmonitor/vllm:infernal-invocation-r18-pcie-autotune-b12x57b5ba9-20260823-exp2 \
MODE=dspark-mtp0 \
BACKEND=b12x-a8-dglin \
ALLREDUCE_MODE=auto \
TP_SIZE=2 \
DCP_SIZE=1 \
MAX_NUM_SEQS=32 \
GRAPH=128 \
MAX_MODEL_LEN=262144 \
MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.95 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r18.yml up -d
```

Use `GPUS=0,1,2,3 TP_SIZE=4` for TP4. Preserve the `/cache` volume because the
calibration result and compiled B12X kernels are keyed to the selected GPUs
and software identity.

For one request with up to 1,048,576 model tokens on TP2, retain the launch
above and override these values:

```bash
MAX_MODEL_LEN=1048576 \
MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.975
```

This profile completed an exact 1,048,002-token prefill with approximately
0.71 GiB of physical memory free per rank near the end of the request. The
margin is part of the qualified contract; do not raise GPU memory utilization
without repeating the long-prefill allocation and health checks.

## Selection Contract

At server startup, `ALLREDUCE_MODE=auto` measures the exact BF16
`[rows, 4096]` operation used by the model. The probe performs 2,048
stabilization replays before timing, checks numerical correctness, runs three
timing trials, and selects B12X only when its median beats NCCL by both one
percent and 0.25 microseconds. The cache fingerprint includes ordered GPU
identity and PCI location, driver, NCCL binary, B12X source, world size,
datatype, hidden size, and measured rows.

The effective policies used by the end-to-end servers were:

| Host | TP | B12X rows | NCCL rows in the measured decode range |
|---|---:|---|---|
| PCIe-switch workstation | 2 | `1-128` | none |
| PCIe-switch workstation | 4 | `1,2,4` | `8,16,24,32` |
| Direct-root-port server | 2 | `1-128` | none |
| Direct-root-port server | 4 | `1,2,4,8,16,24` | `32` |

This result demonstrates why a static topology label is insufficient. The
selected crossover depends on world size, row count, GPU model, host, driver,
and collective implementation. Unsupported shapes use NCCL.

The MoE policy recognizes the DeepSeek-V4 TP2 routed-expert contract
(`E=256`, `K=4096`, rank-local intermediate size `1024`, top-k 6) and uses the
measured persistent grid for 25 through 192 routed rows. Shapes outside that
contract retain the generic B12X policy. TP4 retains its independently
qualified launch ladder.

## Measurement Contract

Both hosts used target-only decode, DCP1, B12X W4A8 plus DGLIN, FP8 compressed
MLA KV, InstantTensor `BUFFERED`, `MAX_NUM_SEQS=32`, graph cap 128,
`MAX_MODEL_LEN=262144`, `MAX_NUM_BATCHED_TOKENS=8192`, and GPU memory
utilization 0.95. Decode cells ran for 30 seconds after a five-second warmup.
Prefill cells used exact 8K, 64K, and 128K token inputs and 15 cold samples.

The workstation GPUs are RTX PRO 6000 Blackwell Workstation Edition devices
behind PCIe switches. The direct-root host uses RTX PRO 6000 Blackwell Server
Edition devices. Host-to-host deltas therefore include GPU SKU and platform
differences and are not a pure PCIe-topology experiment.

## Sustained Decode

Aggregate output throughput in tokens per second:

| Host | TP | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCIe-switch workstation | 2 | 150.4 | 245.2 | 397.9 | 583.7 | 881.2 | 1,315.1 |
| PCIe-switch workstation | 4 | 182.6 | 309.1 | 513.5 | 736.3 | 1,166.4 | 1,791.8 |
| Direct-root-port server | 2 | 130.4 | 210.9 | 347.0 | 515.3 | 782.4 | 1,182.4 |
| Direct-root-port server | 4 | 149.8 | 258.8 | 437.6 | 670.0 | 1,027.3 | 1,593.6 |

## Cold Prefill

Input throughput in tokens per second:

| Host | TP | 8K | 64K | 128K |
|---|---:|---:|---:|---:|
| PCIe-switch workstation | 2 | 14,741 | 13,865 | 12,781 |
| PCIe-switch workstation | 4 | 16,865 | 15,941 | 14,657 |
| Direct-root-port server | 2 | 13,524 | 13,483 | 12,354 |
| Direct-root-port server | 4 | 16,263 | 15,408 | 14,119 |

## Backend Controls

The controls below ran on the PCIe-switch workstation with target-only decode
and `MAX_NUM_BATCHED_TOKENS=4096`. The Lucifer control used the same calibrated
B12X all-reduce, so the comparison isolates model compute dispatch rather than
collective selection.

| Backend | TP | C1 | C2 | C4 | C8 | C16 | C32 | 8K prefill | 64K prefill | 128K prefill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Topology-calibrated B12X | 2 | 150.4 | 245.4 | 396.1 | 582.7 | 875.8 | 1,309.2 | 13,189 | 13,584 | 12,280 |
| Infernal Invocation r18 B12X | 2 | 146.4 | 238.6 | 389.6 | 566.0 | 801.7 | 1,224.5 | 10,901 | 13,351 | 12,304 |
| Lucifer compute plus calibrated B12X all-reduce | 2 | 132.5 | 211.1 | 351.8 | 575.4 | 851.6 | 1,237.6 | 12,887 | 9,418 | 7,709 |
| Infernal Invocation r18 B12X | 4 | 182.4 | 309.6 | 513.9 | 728.1 | 1,132.8 | 1,727.0 | 9,503 | 12,428 | 12,578 |
| Lucifer compute plus calibrated B12X all-reduce | 4 | 149.9 | 259.1 | 437.7 | 718.4 | 1,162.3 | 1,743.5 | 15,028 | 15,019 | 13,595 |

The topology-calibrated B12X image with the 8,192-token prefill envelope beats
the Lucifer control in every measured decode and prefill cell. The table keeps
the 4,096-token controls visible so their different prefill envelope is not
mistaken for an identical scheduler configuration.

## TP2 KV Capacity

`MAX_NUM_BATCHED_TOKENS` controls both prefill work per scheduler step and the
memory admission contract. The asynchronous V2 scheduler permits two batches
in flight, so its unsettled-token bound is:

```text
max_in_flight_tokens = 2 * MAX_NUM_BATCHED_TOKENS
```

DeepSeek-V4 combines full-attention, sliding-window, compressed-MLA, and
indexer cache groups. Sliding-window groups reserve their window plus the
unsettled-token bound for every admitted request. Increasing the batch limit
from 4,096 to 8,192 therefore raises activation memory and doubles the
transient token reservation from 8,192 to 16,384; it is not merely an extra
8,192-token tensor.

| TP2 profile | Async scheduler | Reported KV tokens | KV memory | Peak activation | C1 | 8K prefill | 64K prefill | 128K prefill |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill throughput | yes | 417,641 | 11.46 GiB | 2.17 GiB | 150.4 | 14,741 | 13,865 | 12,781 |
| Resident capacity | yes | 804,694 | 12.54 GiB | 1.49 GiB | 150.4 | 13,189 | 13,584 | 12,280 |
| Synchronous diagnostic | no | 735,116 | 11.46 GiB | 2.17 GiB | 130.9 | 13,333 | 13,772 | 12,715 |

The resident-capacity profile uses `MAX_NUM_BATCHED_TOKENS=4096`. It increases
reported KV capacity by 387,053 tokens, or 92.7 percent, while C1 remains
150.4 tok/s and C2 through C32 remain within 0.7 percent of the prefill
throughput profile. Cold prefill is 10.5 percent slower at 8K, 2.0 percent
slower at 64K, and 3.9 percent slower at 128K.

### One-million-token profile

The 1,048,576-token qualification used GPU memory utilization 0.975. Reported
KV tokens are a group-aware effective capacity calculated by vLLM. DeepSeek-V4
has bounded sliding-window and indexer groups plus full-context groups. Fixed
per-request bounded-group storage is amortized over a longer configured model
length, so this number is not a count of uniform physical token slots and does
not raise the maximum length of one request above `MAX_MODEL_LEN`.

| Batched-token limit | Reported effective KV tokens | Maximum 1M concurrency | C1 | 8K prefill | 256K prefill | 1,048,002-token prefill |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 1,227,878 | 1.17x | 147.9 | 14,590 | 10,936 | 5,723 |
| 4,096 | 2,110,804 | 2.01x | 147.7 | 12,411 | 10,755 | 5,688 |
| 2,048 | 2,835,105 | 2.70x | 147.5 | not measured | 7,094 | 5,134 |

The 8,192-token profile is qualified for one 1M request and preserves the best
measured short-prefill throughput. The 4,096-token result provides enough
reported aggregate capacity for approximately two 1M requests, but two
concurrent 1M requests were not exercised; per-request bounded groups make the
exact admission result workload-dependent. The 2,048-token setting is a
capacity diagnostic, not a recommended serving profile, because its 256K and
1M input throughput is substantially lower.

Disabling async scheduling is not a recommended capacity profile. It recovers
76 percent more reported KV tokens than the prefill throughput profile but
reduces decode by 5.1 to 13.0 percent across C1 through C32 and provides less
capacity than the asynchronous 4,096-token profile.

Additional capacity controls have different tradeoffs:

- Raising `GPU_MEMORY_UTILIZATION` allocates more resident KV but reduces the
  margin for long-prefill activations, JIT workspaces, and communication
  buffers. The documented 1M profile qualifies 0.975; 0.98 would leave only
  approximately 0.23 GiB per rank and is not qualified.
- Lowering the graph cap can recover a small amount of memory when the cap
  exceeds scheduler-reachable rows. The measured graph allocation was only
  0.12 GiB per TP2 rank, so it cannot explain the capacity difference.
- Lowering `MAX_NUM_SEQS` reduces scheduler and graph reachability but does not
  replace the transient-token reduction from a smaller batch limit.
- Native vLLM KV offload and LMCache increase effective history capacity by
  moving blocks out of GPU memory. They do not increase resident GPU KV and
  add a transfer tier.

## Long-context Prefill Analysis

The 1M throughput drop is caused primarily by work that scales with the
already-cached context. It is not caused by routing large prefill collectives
through B12X instead of NCCL.

An explicit NCCL control used the same TP2, 1M model-length, 8,192-token batch,
0.975 memory-utilization, model, and image configuration. `ALLREDUCE_MODE=nccl`
disabled custom all-reduce and the runtime log confirmed `PYNCCL` as the only
TP and EP all-reduce backend.

| Collective path | C1 | 8K prefill | 256K prefill | 1,048,002-token prefill |
|---|---:|---:|---:|---:|
| B12X | 147.9 | 14,590 | 10,936 | 5,723 |
| NCCL | 139.6 | 13,812 | 10,560 | 5,647 |
| NCCL relative to B12X | -5.6% | -5.3% | -3.4% | -1.3% |

The B12X entries and the NCCL 256K and 1M entries are single exact-token
samples. The NCCL 8K entry is the median of ten cold samples. Independent
15-sample 8K controls measured B12X at 14,741 tok/s with a 262K model-length
profile. NCCL did not improve any measured point, so a large-message NCCL
cutoff cannot recover the context-dependent prefill loss.

Rank 0 Torch traces captured four 8,192-token scheduler steps at 32K context
and four steps during a 1M request at approximately 850K accumulated context.
The table reports summed CUDA kernel event durations. Concurrent streams can
overlap, and B12X flag-kernel duration includes time waiting for peer
publication; these values identify scaling work and are not additive wall
time.

| CUDA work over four scheduler steps | 32K context | Approx. 850K context | Growth |
|---|---:|---:|---:|
| All kernel events | 2,553.4 ms | 8,962.1 ms | 3.51x |
| Sparse indexer score scan plus top-k | 91.6 ms (3.6%) | 5,027.5 ms (56.1%) | 54.9x |
| Sparse MLA prefill attention | 299.7 ms (11.7%) | 2,053.9 ms (22.9%) | 6.85x |
| Routed-expert MoE | 487.5 ms (19.1%) | 551.0 ms (6.1%) | 1.13x |
| B12X all-reduce flag plus add kernels | 753.8 ms (29.5%) | 344.3 ms (3.8%) | 0.46x |
| GPU memcpy events | 427.0 ms | 427.6 ms | 1.00x |
| GPU memcpy bytes | 32.6266 GiB | 32.6265 GiB | 1.00x |

The sparse indexer scans an expanding key history for each query chunk. Its
score and top-k kernels account for most of the additional late-context work;
sparse MLA attention is the second scaling component. All-reduce launch count
and GPU-copy duration remain constant across the captures. Further 1M prefill
work should optimize or reduce the indexer scan rather than switch the TP2
collective to NCCL.

Trace identities:

- 32K rank-0 trace SHA-256:
  `cf710d2d3a65417cbd0b6e6dce3a8fe037315f85b36547b437a03453206146ff`
- Approximate-850K rank-0 trace SHA-256:
  `b2f7f051e81ad8e8c39248f40e46d3a31612fa320af8b9b0f18a9fa0ef4d6cdd`
- Parsed profile summary SHA-256:
  `026816ad136d9d79adb8b00d414ea77907c17f0584817d841cc6bd446feaeef4`

## Validation

| Gate | Result |
|---|---|
| B12X focused unit suite | 276 passed, 1 skipped |
| TP2 MoE launch-policy unit suite | 32 passed |
| vLLM helper and custom all-reduce suite | 34 passed, 1 skipped |
| Formatting and static checks | Ruff, formatter, and whitespace checks passed |
| All-reduce correctness | Every timed calibration row matched the NCCL reference |
| End-to-end health | TP2 and TP4 decode and prefill matrices completed on both hosts |
| TP2 1M allocation | Exact 1,048,002-token prefill completed at batched-token limits 2,048, 4,096, and 8,192 |
| TP2 collective control | Explicit NCCL completed C1 and exact 8K, 256K, and 1M prefills; every result was slower than B12X |
| Context-scaling profile | Clean four-step rank-0 traces captured at 32K and approximately 850K context |

## Qualification Limits

- **Implemented:** topology- and shape-keyed B12X/NCCL all-reduce selection,
  TP2 remote-push all-reduce, and DeepSeek-V4 TP2 routed-expert launch policy.
- **Research-only:** the immutable image and throughput measurements on the
  two documented hosts.
- **Unsupported by this evidence:** MTP or DSpark, DCP greater than one, TP8,
  TP16, model-quality claims, and unattended production deployment.
- The calibration cache must not be copied between different ordered GPU sets
  or software identities. A fingerprint mismatch causes a fresh probe.
