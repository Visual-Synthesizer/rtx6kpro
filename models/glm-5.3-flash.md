# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke in the
<a href="https://discord.com/channels/1466898002793857221/1476263308242714718/1543077243398393927">community Discord</a>.</em></p>

This page specifies the qualified GLM-5.3-Flash deployment for four NVIDIA RTX
PRO 6000 Blackwell GPUs. The serving artifact is Jovian Judgement Community
`20260902-r15`. It supports ordinary decode, three-token Multi-Token
Prediction (MTP), and a seven-token DFlash2 draft. The artifact does not expose
the LMCache sidecar launcher; use the historical LMCache artifact documented
below only when external DRAM or filesystem prefix storage is required.

The commands use Hugging Face repository names and named Docker volumes. They
do not require checkpoint paths or source-code bind mounts.

## Status

| Field | Value |
|---|---|
| Runtime status | **qualified** for Tensor Parallelism 4 (TP4) with Decode Context Parallelism 1 (DCP1) in all three serving modes |
| DCP2 | **implemented**; not independently performance-qualified for this artifact |
| DCP4 full compressed-key/value prefill | **implemented**; the target path was qualified in an earlier source composition and is not independently requalified here |
| Hardware | four RTX PRO 6000 Blackwell Workstation Edition GPUs; reported performance uses stock 14,001 MHz maximum VRAM clock |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Target update policy | resolve Hugging Face `main` at startup unless `MODEL_REVISION` is set |
| Target experts | ModelOpt NVFP4 with B12X 4-bit weights and 4-bit activations |
| Target KV cache | FP8 compressed Multi-head Latent Attention |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2` |
| DFlash2 update policy | resolve Hugging Face `main` at startup unless `DFLASH_MODEL_REVISION` is set |
| DFlash2 weights | offline-serialized ModelOpt MXFP8; no online weight quantization |
| Cache page geometry | separate 512-token target and recurrent-state pages |
| Scheduler | 4,096 maximum batched tokens; concurrent-prefill interval 8 |
| CUDA graphs | target and speculative decode captured; Gated Delta Network prefill eager |
| LMCache | **unsupported by this artifact's launcher**; the historical `20260901-r12` artifact remains qualified for 4,096-token DRAM/filesystem objects |
| Root filesystem | two Docker layers; compatible with standard overlay2 depth limits |
| Qualification date | 2026-09-02 |

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260902-r15
voipmonitor/vllm@sha256:d9ca4c299fdb8e6176d70f36c8f98a687474eef807b7a71fe6fee9f154208fe0
```

The local qualified image ID is
`sha256:a54046ef6be79da889c3d19e0f4b2a8d50109446ed61ee065756aba596026aa0`.
The embedded source-lock SHA-256 is
`bff19e7c405a5346e4b701c874411ad56564a416038a88d78aa3df5c0277e982`.
The Docker digest fixes the runtime. Model repository names follow their
`main` branches unless the optional revision variables are set.

## Source contract

The vLLM package is based on `local-inference-lab/vllm`
`dev/jovian-judgement@9c4dd05487629eccb26d7166459867a3db9b099f` and contains
the following open, non-draft pull requests:

| Pull request | Resulting behavior |
|---|---|
| [vLLM #552](https://github.com/local-inference-lab/vllm/pull/552) | Fuses GLM query scaling into the Walsh-Hadamard-transform/FP8 quantization kernel. |
| [vLLM #553](https://github.com/local-inference-lab/vllm/pull/553) | Permits expandable segments for engine-driven external-KV transfers. |
| [vLLM #554](https://github.com/local-inference-lab/vllm/pull/554) | Shuts multiprocess workers down after imported KV mappings are unregistered. |
| [vLLM #555](https://github.com/local-inference-lab/vllm/pull/555) | Restricts manually enabled shareable cuMem allocation to KV-cache storage. |
| [vLLM #557](https://github.com/local-inference-lab/vllm/pull/557) | Exports exact recurrent-cache boundaries to external stores. |
| [vLLM #558](https://github.com/local-inference-lab/vllm/pull/558) | Fuses the supported DFlash2 two-tap grouped convolution. |
| [vLLM #573](https://github.com/local-inference-lab/vllm/pull/573) | Matches probabilistic DFlash proposal temperature, top-k, and top-p constraints. |
| [vLLM #571](https://github.com/local-inference-lab/vllm/pull/571) | Reuses normalized constraint probabilities in standard rejection sampling. |
| [vLLM #572](https://github.com/local-inference-lab/vllm/pull/572) | Uses FlashInfer top-p renormalization under `VLLM_USE_FLASHINFER_SAMPLER`. |
| [vLLM #576](https://github.com/local-inference-lab/vllm/pull/576) | Prefetches upcoming projection weights into the 128 MB SM120 L2 cache during reduction windows. |

The reproducible vLLM integration commit is
`c28bb6e2dc959a75742a44ab08cc4a340ff4cd3c`, its source tree is
`c7523414ba9aa8d07f63b497eddfc1fc45faccc1`, and the installed `vllm/`
package tree is `7e389bd1850896720fd7b29ef43a153d998718b1`.

The B12X package is based on `local-inference-lab/b12x`
`master@da8f2cb3218bbdb99deec979c6f503ff09d0cb89` and contains:

| Pull request | Resulting behavior |
|---|---|
| [B12X #260](https://github.com/local-inference-lab/b12x/pull/260) | Reduces paged top-k-512 shared storage and omits an unused indices-only score write. |
| [B12X #267](https://github.com/local-inference-lab/b12x/pull/267) | Overlaps native NVFP4 M=1 FC2 row-pair loads. |
| [B12X #268](https://github.com/local-inference-lab/b12x/pull/268) | Fuses W4A4 FC1 while preserving swapped gated-weight scratch capacity. |
| [B12X #269](https://github.com/local-inference-lab/b12x/pull/269) | Selects measured recurrent GDN tiles for GLM-5.3 graph capacities. |
| [B12X #270](https://github.com/local-inference-lab/b12x/pull/270) | Selects measured multipath-hyperconnection partial grouping. |
| [B12X #272](https://github.com/local-inference-lab/b12x/pull/272) | Elides MXFP8 scale-padding fills that no supported consumer reads. |
| [B12X #278](https://github.com/local-inference-lab/b12x/pull/278) | Uses fill-eliding quantization in packed MXFP8 linear operations. |
| [B12X #273](https://github.com/local-inference-lab/b12x/pull/273) | Specializes eight-row GLM-5.3 route preparation and dispatch. |
| [B12X #280](https://github.com/local-inference-lab/b12x/pull/280) | Parallelizes M8 route quantization over `(token, route)`. |
| [B12X #279](https://github.com/local-inference-lab/b12x/pull/279) | Adds optional dependent launch between PCIe one-shot publication and MHC consumption. |

The reproducible B12X integration commit is
`a23edac6b335b0c3391d7523a5aeec6e52802882`, its source tree is
`7f36af8e3613c8228c7e22a509c40a5911d15479`, and the installed `b12x/`
package tree is `2a69d69e890006e931d535aea52891e3784b0b55`.

The complete merge order, dependencies, and qualification evidence are tracked
in [vLLM issue #500](https://github.com/local-inference-lab/vllm/issues/500).

### Historical `20260901-r12` LMCache source contract

The vLLM runtime preserves the R11 GLM-5.3 package at
`a02841bcf218b067ca352d97be514e0e8fedb896` and adds six open pull-request
heads. Their upstream base is `local-inference-lab/vllm` branch
`dev/jovian-judgement` at
`54f6e9826c20ef06ed65d838c0ad497ad0abdecf`.

| Pull request | Resulting behavior |
|---|---|
| [vLLM #550](https://github.com/local-inference-lab/vllm/pull/550) | Removes redundant B12X sparse-decode metadata work and emits physical cache rows directly. |
| [vLLM #552](https://github.com/local-inference-lab/vllm/pull/552) | Fuses and retunes GLM query scaling inside the fast Walsh-Hadamard-transform and FP8 quantization kernel. |
| [vLLM #553](https://github.com/local-inference-lab/vllm/pull/553) | Allows expandable CUDA segments when an external KV connector uses the engine-driven transfer mode. |
| [vLLM #554](https://github.com/local-inference-lab/vllm/pull/554) | Stops multiprocess workers cleanly when the supervised server exits. |
| [vLLM #555](https://github.com/local-inference-lab/vllm/pull/555) | Restricts the manual shareable-cuMem allocator to KV-cache allocation instead of all CUDA allocations. |
| [vLLM #557](https://github.com/local-inference-lab/vllm/pull/557) | Exposes exact recurrent-cache retention boundaries to external stores, schedules those boundaries deterministically, and guards external block allocation from negative requests. |

The reproducible vLLM composition commit is
`7d8a09c42c7ba743b9e936562aa9205f9d0fda9d`, its source tree is
`237bc2b7bb297ab69b83704fad0b9f6628bfcde8`, and the installed `vllm/`
package tree is `12a88973c99db8b51937ac9b4f81dac0a5a6706b`.

The B12X runtime is composed from `local-inference-lab/b12x`
`master@139e04048bc3bb4f7210c99e7184d8d2f0e345e7` plus these open
pull-request heads:

| Pull request | Resulting behavior |
|---|---|
| [B12X #260](https://github.com/local-inference-lab/b12x/pull/260) | Increases top-k-512 candidate capacity and omits unused terminal score writes while preserving indices-only correctness. |
| [B12X #267](https://github.com/local-inference-lab/b12x/pull/267) | Overlaps native NVFP4 M=1 FC2 row-pair loads for the 288-expert GLM-5.3 geometry. |
| [B12X #268](https://github.com/local-inference-lab/b12x/pull/268) | Fuses native W4A4 FC1 projections while preserving scratch capacity for swapped gated weights. |
| [B12X #269](https://github.com/local-inference-lab/b12x/pull/269) | Selects profiled recurrent GDN tiles for the GLM-5.3 serving graph capacities. |
| [B12X #270](https://github.com/local-inference-lab/b12x/pull/270) | Selects the profiled multipath-hyperconnection partial grouping for GLM-5.3 decode. |

The reproducible B12X composition commit is
`d064ca4cc3aa25292f1a8756fa51b38134d1df84`, its source tree is
`aace94c2fcc0657c1aa255e9480277c8f30240fc`, and the installed `b12x/`
package tree is `c9384d70bd581897a16425efd43fa79374c589e3`.

The optional LMCache runtime is composed from
`local-inference-lab/LMCache` at
`801b6ce335a46628bd87b70b8c1c263f45a380f3`:

| Pull request | Resulting behavior |
|---|---|
| [LMCache #33](https://github.com/local-inference-lab/LMCache/pull/33) | Shares KV-cache cuMem allocations with the LMCache sidecar through CUDA IPC and closes every imported and exported allocation through an explicit ownership lifecycle. |
| [LMCache #34](https://github.com/local-inference-lab/LMCache/pull/34) | Uses vLLM scheduler block tables and exact recurrent boundaries as store sources, assigns every store its own completion identity, and pins source blocks until all TP ranks finish the asynchronous store. |

The installed LMCache package tree is
`2136aada94fd5780d50fa84baaad4f5fb709305c`. Recurrent-cache objects require
eager source admission; LMCache lazy-store mode is unsupported for this model
because it cannot guarantee that recurrent source states remain live.

## Runtime backends

| Operation | Selected implementation |
|---|---|
| Target sparse attention and C4 indexer | B12X |
| Target GDN prefill | FlashKDA recurrent checkpoints |
| Target GDN decode | B12X live-tensor KDA when eligible; Triton fallback retained |
| Target routed experts | B12X NVFP4 W4A4 |
| Target linear layers | B12X |
| Tensor-parallel all-reduce | B12X PCIe first; PyNCCL outside the B12X dispatch range |
| MTP attention | B12X |
| MTP MXFP8 experts | Humming |
| DFlash2 MXFP8 linear layers | `B12xMxfp8LinearKernel` |
| DFlash2 fused context key/value projection | `B12xMxfp8LinearKernel` |
| DFlash2 local attention | FlashAttention 2 |
| Sampling | FlashInfer |
| Prefix reuse before external lookup | vLLM Automatic Prefix Caching |
| External prefix cache | unsupported by this artifact's launcher |

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths in this serving contract.

## Start a DCP1 server

Pull the qualified image, set the four physical GPUs, and select one serving
mode:

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260902-r15
GPU_DEVICES=0,1,2,3
docker pull "$IMAGE"
```

```bash
# Ordinary decode without speculative tokens.
NAME=jovian-judgement-nospec-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=0)
```

```bash
# Three-token built-in MTP.
NAME=jovian-judgement-mtp3-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=3)
```

```bash
# DFlash2 with its trained seven-draft-token configuration.
NAME=jovian-judgement-dflash2-dcp1
MODE_ARGS=(
  -e SPECULATOR=dflash2
  -e NUM_SPECULATIVE_TOKENS=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2
)
```

Run the selected mode. Automatic Prefix Caching is enabled by default inside
the vLLM process; this artifact does not start an external LMCache sidecar.

```bash
docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-vllm-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e PORT=5001 \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_SEQS=16 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=8 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
  -e GPU_MEMORY_UTILIZATION=0.90 \
  -e B12X_PCIE_ALLREDUCE=1 \
  -e NCCL_MIN_NCHANNELS=32 \
  -e NCCL_MAX_NCHANNELS=32 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e OMP_NUM_THREADS=2 \
  "${MODE_ARGS[@]}" \
  "$IMAGE"
```

DFlash2 reserves its draft-query rows separately, so the launcher keeps the
target work budget at 4,096 tokens while raising only the internal input-row
capacity. No additional scheduler argument is required.

For DCP2, replace `DCP=1` with `DCP=2`. For DCP4 full-CKV prefill, use:

```bash
  -e DCP=4 \
  -e DCP_CKV_GATHER=1
```

The launcher enables full-CKV gathering automatically when DCP is greater than
one. The explicit variable documents the selected behavior.

For reproducible deployments, add immutable `MODEL_REVISION` and
`DFLASH_MODEL_REVISION` values. Omit them to receive checkpoint updates from
the Hugging Face `main` branches.

## Verify startup

```bash
curl -fsS http://127.0.0.1:5001/health

docker logs "$NAME" 2>&1 | grep -E \
  'B12X PCIe|B12xMxfp8|HUMMING|split GLM-5.3 cache pages|l2_prefetch|Graph capturing finished|Application startup complete'

curl -fsS http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":64}'
```

Common markers include B12X NVFP4 MoE, B12X PCIe all-reduce, 512-token target
plus recurrent pages, and successful L2-prefetch plan construction on each
rank. MTP adds Humming for its MXFP8 experts. DFlash2 adds
`B12xMxfp8LinearKernel` and FlashAttention 2.

## Stock-clock TP4/DCP1 qualification

Status: **qualified**. The comparison used physical GPUs 4-7 at stock clocks,
4,096 maximum batched tokens, separate 512-token target and recurrent pages,
FP8 target KV, B12X target attention/linear/MoE, B12X PCIe all-reduce, 32 NCCL
channels, captured target/speculative decode graphs, and revision-isolated JIT
caches. C1 and C8 denote request concurrency one and eight.

Speculative decode reports target/verifier forward passes per second because
accepted length is stochastic. Each artifact received one warmup and three
30-second measured decode runs. The DFlash2 result includes a second
three-run candidate block because the first C8 block showed run-to-run
variation.

| Serving mode | C1 decode | C8 decode | 32k prefill |
|---|---:|---:|---:|
| No speculation | 161.541 output tok/s | 730.683 output tok/s | 14,646 prompt tok/s |
| MTP with three draft tokens | 101.612 target forwards/s | 360.829 target forwards/s | 14,284 prompt tok/s |
| DFlash2 with seven draft tokens | 88.188 target forwards/s | 278.454 target forwards/s | 14,368 prompt tok/s |

Relative to reference image
`sha256:85607a7ff8b6321364c8cfbff8dcc5b368e29aceb0ebe961f76dc1b817beae6b`,
no-speculation decode improved 8.48% at C1 and 4.46% at C8; MTP
target-forward throughput improved 4.10% and 2.37%; DFlash2 improved 3.96% and
0.18%. All three 32k-prefill changes were within 0.24% of the reference and no
prefill regression was reproduced.

Exact log-probability parity is unsupported as a correctness oracle for this
B12X configuration. Five identical no-speculation requests to the reference
artifact retained the same greedy top-1 token but varied its reported log
probability from -0.66834 to -0.78686. The qualified artifact retained the same
greedy token.

## Historical `20260901-r12` LMCache qualification

Status: **qualified**. The qualification used the published R12 source
composition on physical GPUs 4–7 with TP4/DCP1, FP8 target KV, 512-token
target and recurrent pages, a 4,096-token target scheduler budget, B12X PCIe
all-reduce, 32 NCCL channels, and a +6000 MHz VRAM offset. LMCache used a
4,096-token object size, 64 GiB DRAM L1, and native-filesystem L2.

`llm-decode-bench` 0.4.30 measured 30-second sustained context-zero decode and
twelve standalone cold 32k-prefill requests over 30 seconds. Sieve uses the
coding prompt `Write a Python script that implements the Sieve of
Eratosthenes.` and reports the median output rate across repeated runs.

| Mode | 32k prefill | CC1 output | Sieve median | Verifier steps/s | Accepted length |
|---|---:|---:|---:|---:|---:|
| No speculation | 15,078 tok/s | 167.5 tok/s | 168.9 tok/s | — | — |
| MTP:3 | 14,788 tok/s | 254.3 tok/s | 333.6 tok/s | 107.09 | 2.37 |
| DFlash2:7 | 14,757 tok/s | 187.4 tok/s | 403.6 tok/s | 88.42 | 2.12 |

The same R12 image with LMCache disabled measured 15,434 tok/s for no-spec
32k prefill and 167.6 tok/s for no-spec CC1 decode. Active external stores
therefore cost 2.31% in the cold-prefill cell and no measurable sustained
decode throughput. Prefix hits amortize that store cost by restoring complete
4,096-token objects.

Correctness qualification compared every live byte in each restored target
attention group and all four recurrent-state groups across all four TP ranks.
It covered no-spec, MTP:3, and DFlash2 at DCP1 and DFlash2 full-CKV at DCP4.
Cold, vLLM APC, LMCache DRAM L1, LMCache native-filesystem L2, and a complete
sidecar-plus-vLLM restart from L2 all restored the expected prefix. The final
Docker digest independently restored 8,192 external tokens from disjoint
source and destination blocks.

## Historical R11 and R8 comparison

Status: **qualified**. R8 and R11 ran sequentially on physical GPUs 8–11 at
stock clocks with the same target and draft revisions, TP4/DCP1, FP8 target KV,
512-token target and recurrent pages, B12X PCIe all-reduce, 32 NCCL channels,
and a 4,096-token scheduler budget.

The exact comparison artifacts are
`voipmonitor/vllm@sha256:827a64ce0cea267aad843b3d521a47d742a6e78b502eaec7c05b4ae8bf403194`
for R8 and
`voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43`
for R11.

`llm-decode-bench` 0.4.30 measured 30-second sustained decode cells. Paired
speculative cells used the same deterministic cache-busting prefix and
tokenized input, because accepted length is prompt-dependent.

| Mode | Context | Concurrency | R8 tok/s | R11 tok/s | Change |
|---|---:|---:|---:|---:|---:|
| No speculation | 0 | 1 | 137.84 | **147.12** | **+6.74%** |
| MTP:3 | 0 | 1 | 228.13 | **251.20** | **+10.11%** |
| DFlash2:7 | 0 | 1 | 185.68 | **192.28** | **+3.55%** |
| DFlash2:7 | 16k | 1 | 190.70 | **199.75** | **+4.74%** |
| DFlash2:7 | 0 | 8 | 688.83 | **729.47** | **+5.90%** |
| DFlash2:7 | 16k | 8 | 700.36 | **728.46** | **+4.01%** |

Acceptance-normalized engine rate isolates target-forward speed from the number
of draft tokens accepted by a particular output trajectory.

| Mode | Context | Concurrency | R8 steps/s | R11 steps/s | Change |
|---|---:|---:|---:|---:|---:|
| MTP:3 | 0 | 1 | 89.78 | **94.20** | **+4.92%** |
| DFlash2:7 | 0 | 1 | 73.85 | **78.17** | **+5.85%** |
| DFlash2:7 | 16k | 1 | 73.53 | **77.55** | **+5.48%** |
| DFlash2:7 | 0 | 8 | 264.88 | **270.69** | **+2.20%** |
| DFlash2:7 | 16k | 8 | 264.36 | **270.89** | **+2.47%** |

Standalone cold-prefill runs issued twelve 32k requests over 30 seconds.
Client throughput is prompt tokens divided by time to first token. Speculative
modes therefore include the work required to emit the first verified token.

| Mode | Prompt tokens | R8 tok/s | R11 tok/s | Change | R11 TTFT |
|---|---:|---:|---:|---:|---:|
| No speculation | 32,320 | 14,550 | **14,572** | **+0.15%** | 2.218 s |
| MTP:3 | 32,321 | 14,210 | **14,228** | **+0.13%** | 2.272 s |
| DFlash2:7 | 32,321 | 14,233 | **14,244** | **+0.08%** | 2.269 s |

The prefill deltas are within run-to-run noise and establish parity rather than
a material speedup. Decode is consistently faster in R11.

### R8 research-profile archive

Status: **research-only**. These measurements preserve the R8 memory-clock and
Sieve coding-prompt qualification; they are not substitutes for the matched
R8/R11 regression cells above. Physical GPUs 4–7 used TP4/DCP1, FP8 target KV,
512-token target and recurrent pages, a 4,096-token scheduler budget, B12X PCIe
all-reduce, and 32 NCCL channels. The overclocked row changed only the GPU
memory-clock offset and returned all offsets to zero after measurement.

| R8 mode | Clock profile | 32k prefill | CC1 output | Sieve median | Engine rate | Accepted length |
|---|---|---:|---:|---:|---:|---:|
| No speculation | stock | 15,549 tok/s | 139.4 tok/s | not measured | — | — |
| MTP:3 | stock | 15,131 tok/s | 228.0 tok/s | 287.87 tok/s | 90.47 steps/s | 2.52 |
| DFlash2:7 | stock | 15,276 tok/s | 185.5 tok/s | 339.73 tok/s | 74.19 steps/s | 2.49 |
| DFlash2:7 | VRAM +6000 | 15,759.6 tok/s | 200.87 tok/s | not measured | 82.91 steps/s | 2.43 |

The Sieve cells generated at most 2,000 tokens for the prompt
`Write a Python script that implements the Sieve of Eratosthenes.`. The result
is prompt-specific because speculative acceptance depends on the generated
trajectory. The same R8 DFlash2 configuration at TP4/DCP4 with full-CKV gather
measured 13,578 prompt tokens/s for a 32,320-token request.

## DFlash2 MXFP8 checkpoint

`local-inference-lab/GLM-5.3-Flash-DFlash2` is an offline MXFP8 conversion of
`incoai/GLM-5.3-Flash-DFlash2` revision
`dc77ff1c99eeb2df044ee3d4f0094eb033fee410`. The published conversion commit
is `aea0ac8a05624512ca9e106c09c16087da998426`.

The converter serialized 47 two-dimensional linear weights as FP8 E4M3 values
with one biased E8M0 `uint8` scale per 32 input values. It preserved 34
nonlinear tensors bit-for-bit. Independent validation reproduced every
quantized value and scale exactly, loaded the checkpoint with B12X MXFP8
kernels, captured target and draft decode graphs, and completed speculative
inference.

The launcher intentionally uses the repository name without a revision suffix.
This causes a restart to pick up a later Hugging Face `main` commit. Pin
`DFLASH_MODEL_REVISION` when exact reproducibility is more important than
automatic checkpoint updates.

## Limitations

- GDN prefill executes eagerly because the backend supports uniform-batch
  decode capture, not full prefill capture.
- Speculative raw output throughput depends on accepted length. Engine steps
  per second is the cleaner runtime-regression metric.
- DCP2 is implemented but has no independent performance qualification for the
  `20260902-r15` artifact.
- The `20260902-r15` launcher does not support LMCache. The historical
  `20260901-r12` LMCache artifact requires eager recurrent-state admission;
  prefixes shorter than its 4,096-token object size remain target-model work.
- Mutable Hugging Face `main` branches can change model behavior without
  changing the Docker digest. Pin revisions for reproducible evaluation.
