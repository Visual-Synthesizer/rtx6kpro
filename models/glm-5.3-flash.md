# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke in the
<a href="https://discord.com/channels/1466898002793857221/1476263308242714718/1543077243398393927">community Discord</a>.</em></p>

This page specifies the qualified GLM-5.3-Flash deployment for four NVIDIA RTX
PRO 6000 Blackwell GPUs. The serving artifact is Jovian Judgement Community
`20260901-r11`. It supports ordinary decode, three-token Multi-Token
Prediction (MTP), and a seven-token DFlash2 draft.

The commands use Hugging Face repository names and named Docker volumes. They
do not require checkpoint paths or source-code bind mounts.

## Status

| Field | Value |
|---|---|
| Runtime status | **qualified** for Tensor Parallelism 4 (TP4) with Decode Context Parallelism 1 (DCP1) in all three serving modes |
| DCP2 | **implemented**; no independent R11 performance qualification |
| DCP4 full compressed-key/value prefill | **qualified** for the GLM-5.3 target path; the R11/R8 performance tables below use DCP1 |
| Hardware | four RTX PRO 6000 Blackwell Workstation Edition GPUs, PCIe 5.0 x16, stock clocks |
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
| Root filesystem | one Docker layer; compatible with standard overlay2 depth limits |
| Qualification date | 2026-09-01 |

## Docker artifact

```text
voipmonitor/vllm:jovian-judgement-community-20260901-r11
voipmonitor/vllm@sha256:93ac5228f1cbde2182ca294d8b479259144742af2756a49ff207dd245429bf43
```

The local qualified image ID is
`sha256:1e3718f0836998e0d5f6e8c4da2bce9c10092301dab10932cfe8b35df0139b3d`.
The embedded source-lock SHA-256 is
`c3e6dcc60c0e668d0d38ce4d59d17af1c4c1e0326e0f8e1b7c6fc4c33fc8c3ac`.
The Docker digest fixes the runtime. Model repository names follow their
`main` branches unless the optional revision variables are set.

## Source contract

The vLLM runtime is composed from
`local-inference-lab/vllm` branch `dev/jovian-judgement` at
`54f6e9826c20ef06ed65d838c0ad497ad0abdecf` plus two open pull-request
heads:

| Pull request | Resulting behavior |
|---|---|
| [vLLM #550](https://github.com/local-inference-lab/vllm/pull/550) | Removes redundant B12X sparse-decode metadata work and emits physical cache rows directly. |
| [vLLM #552](https://github.com/local-inference-lab/vllm/pull/552) | Fuses and retunes GLM query scaling inside the fast Walsh-Hadamard-transform and FP8 quantization kernel. |

The reproducible vLLM composition commit is
`a02841bcf218b067ca352d97be514e0e8fedb896`, its source tree is
`7c51a8c7958780895a1f8f8d74de0908aec97849`, and the installed `vllm/`
package tree is `75316c408d3fea4306518402d3027d06f4352806`.

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

DeepGEMM and TileLang are installed dependencies but are not selected for the
target, MTP, or DFlash2 hot paths in this serving contract.

## Start a DCP1 server

Set the four physical GPUs and select one mode:

```bash
IMAGE=voipmonitor/vllm:jovian-judgement-community-20260901-r11
GPU_DEVICES=0,1,2,3
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

Run the selected mode:

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
  'B12X PCIe|B12xMxfp8|HUMMING|split GLM-5.3 cache pages|Graph capturing finished|Application startup complete'

curl -fsS http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":64}'
```

Common markers include B12X NVFP4 MoE, B12X PCIe all-reduce, and 512-token
target plus recurrent pages. MTP adds Humming for its MXFP8 experts. DFlash2
adds `B12xMxfp8LinearKernel` and FlashAttention 2.

## R11 performance and R8 comparison

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
`bf582e4eacc1810f76656d1811693ff6c6737d2a`. The published conversion commit
is `cc006ae7801b80c8f845aa5990d183aaa4bd1cff`.

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
- DCP2 is implemented but has no independent R11 performance table.
- Mutable Hugging Face `main` branches can change model behavior without
  changing the Docker digest. Pin revisions for reproducible evaluation.
