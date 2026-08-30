# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke in the
<a href="https://discord.com/channels/1466898002793857221/1476263308242714718/1543077243398393927">community Discord</a>.</em></p>

This page is the stable deployment and performance reference for
GLM-5.3-Flash on RTX PRO 6000 Blackwell. The qualified serving artifact is
Jovian Judgement Community DFlash2 — 20260830-r7.

Terminology used below:

- The vLLM inference engine `vLLM` integrates the B12X kernel/backend stack
  `B12X`.
- 4-bit floating-point `FP4`, NVIDIA FP4 `NVFP4`, 8-bit floating-point `FP8`,
  and Microscaling FP8 `MXFP8` name numeric formats.
- Multi-Token Prediction `MTP`, Decode Context Parallelism `DCP`, Multi-head Latent Attention `MLA`,
  Mixture of Experts `MoE`, and Time To First Token `TTFT` name model or serving
  operations.
- Compute Unified Device Architecture `CUDA`, NVIDIA Collective Communications Library `NCCL`,
  Peripheral Component Interconnect Express `PCIe`, and Streaming Multiprocessor 120 `SM120`
  name the runtime and hardware interfaces.

The runbook serves `local-inference-lab/GLM-5.3-Flash-NVFP4` on four NVIDIA
RTX PRO 6000 Blackwell GPUs. The Jovian Judgement Community DFlash2 image
supports ordinary decode, three-token Multi-Token Prediction (MTP), and a
seven-token DFlash2 draft loaded from
`local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8`.

The commands use Hugging Face model names and named Docker volumes. They do not
require local checkpoint paths or source-code bind mounts.

## Status

| Field | Value |
|---|---|
| Runtime status | **qualified** for Tensor Parallelism 4 (TP4) with Decode Context Parallelism 1 (DCP1) in all three serving modes |
| Additional qualification | **qualified** for TP4/DCP4 DFlash2 prefill with full compressed-key/value (CKV) gathering |
| Hardware | four RTX PRO 6000 Blackwell Workstation Edition GPUs, PCIe 5.0 x16, stock clocks |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Target routed experts | ModelOpt NVFP4, B12X 4-bit-weight/4-bit-activation (W4A4) |
| Target key-value cache | FP8 compressed Multi-head Latent Attention (MLA) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8@d982b94051c63a31f74a2331c714635de0d5cda9` |
| DFlash2 weights | pre-serialized ModelOpt MXFP8; no online weight quantization |
| Cache page geometry | independent 512-token target and recurrent-state pages |
| Scheduler limit | `MAX_NUM_BATCHED_TOKENS=4096` |
| CUDA graphs | target and speculative decode are captured; Gated Delta Network (GDN) prefill is eager |
| Qualification date | 2026-08-30 |

## Docker artifact

Use the digest for a byte-identical deployment:

```text
voipmonitor/vllm:jovian-judgement-community-dflash2-20260830-r7
voipmonitor/vllm@sha256:3a7cbc5833ed3d5390c5de2ec3c5a2737761a175123262cfdfd7b38638e1d5e6
```

The qualified local image ID is
`sha256:eaba6a8a03f0eef449388658e89aaa1e00a564b8479e131eec2904040c0b7664`.
The embedded source-lock SHA-256 is
`30da939ce855fd059c866b7f177e5f36abad73ad0618f14fc0cf49f50d58b80a`.

## Source contract

The vLLM package tree is
`54fee66bc5cdba569493b8685fa262494ca1fdf0`. It is composed from
`local-inference-lab/vllm` branch `dev/jovian-judgement` at
`0b67266a0f37d6146a8403fb8482403c62f412d5` and the following non-draft pull
request heads:

| Pull request | Resulting behavior |
|---|---|
| [vLLM #515](https://github.com/local-inference-lab/vllm/pull/515) `3bcb90163d9f` | Retains CUDA-graph profiling resources until teardown. |
| [vLLM #516](https://github.com/local-inference-lab/vllm/pull/516) `db0f14444e8c` | Makes the B12X profiling warmup lifetime-safe. |
| [vLLM #517](https://github.com/local-inference-lab/vllm/pull/517) `92d807af333b` | Gathers full C4 CKV for DCP prefill. |
| [vLLM #530](https://github.com/local-inference-lab/vllm/pull/530) `3609a3db4986` | Uses FlashKDA recurrent checkpoints for GLM-5.3 prefill. |
| [vLLM #531](https://github.com/local-inference-lab/vllm/pull/531) `b8edca554d21` | Reuses immutable B12X C4 indexer plans. |
| [vLLM #532](https://github.com/local-inference-lab/vllm/pull/532) `71054201ae23` | Parallelizes C4 pool writes and bounds visible pages. |
| [vLLM #533](https://github.com/local-inference-lab/vllm/pull/533) `d6ace9116f9c` | Preserves replicated DFlash2 cache geometry under DCP. |
| [vLLM #535](https://github.com/local-inference-lab/vllm/pull/535) `5f8e00d6c33a` | Separates target and recurrent cache pages at 512 tokens. |
| [vLLM #536](https://github.com/local-inference-lab/vllm/pull/536) `9f27029f55fd` | Checkpoints target GDN state for MTP prefill. |
| [vLLM #537](https://github.com/local-inference-lab/vllm/pull/537) `236032509531` | Compacts MTP prefill outputs before Mixture-of-Experts (MoE) compute. |
| [vLLM #539](https://github.com/local-inference-lab/vllm/pull/539) `d59cea8e55f6` | Preserves the embedding of a valid MTP token at position zero. |

The reproducible vLLM composition commit is
`voipmonitor/vllm@bb3f9c3a00af522af93478645ad391f51eb0225b` with tree
`9cd4481415a8bae84e49f6e3da9f0e47c5ed0889`. Pull request #513 is not part of
this package tree.

The B12X package tree is
`6de9871d15dab093340695518fec0f744289e676`. It is composed from B12X
`master@fc1d4b68f7a5b0cfdb88bf06abccd869f5c589d5` and two non-draft
performance pull requests:

| Pull request | Scope | Qualified kernel result |
|---|---|---:|
| [B12X #259](https://github.com/local-inference-lab/b12x/pull/259) `e5509cc95b8f` | Selects a wider SM120 TF32 multi-head-connection projection for hidden size 4096 and 2,304–3,583 prefill rows. This operation is shared by all three target-serving modes. | 31.67% projection-throughput increase at 3,072 rows |
| [B12X #260](https://github.com/local-inference-lab/b12x/pull/260) `b68c197262d5` | Increases device-bounded shared candidate capacity and omits unused score output in the paged C4 top-k selector. This operation is shared by all three target-prefill modes. | 2.01% selector-throughput increase at 4,080 query rows |

Those percentages qualify the named fixed-work kernels, not end-to-end server
throughput. The reproducible B12X composition commit is
`voipmonitor/b12x@6255090a03b12c3f7d552102a02fac0b542fb8c9` with tree
`0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad`.

## Runtime backends

| Operation | Selected implementation |
|---|---|
| Target sparse MLA attention and C4 indexer | B12X |
| Target GDN prefill | FlashKDA recurrent checkpoints |
| Target GDN decode | B12X live-tensor KDA when eligible; automatic resolver retains the Triton fallback |
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

`CUDAGRAPH_MODE=FULL` is requested. The GLM GDN backend supports uniform-batch
decode capture but not full prefill capture, so vLLM resolves target execution
to `FULL_DECODE_ONLY`. MTP and DFlash2 decode graphs are captured. Prefill is
eager.

## Start one of the three DCP1 modes

Set the immutable image, the four desired physical GPUs, and a unique container
name. The qualification host used GPUs 4, 5, 6, and 7; `0,1,2,3` below is a
portable four-GPU example.

```bash
IMAGE=voipmonitor/vllm@sha256:3a7cbc5833ed3d5390c5de2ec3c5a2737761a175123262cfdfd7b38638e1d5e6
GPU_DEVICES=0,1,2,3
```

Select exactly one mode:

```bash
# Ordinary decode without speculative tokens.
NAME=jovian-judgement-nomtp-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=0)
```

```bash
# Three-token built-in MTP.
NAME=jovian-judgement-mtp3-dcp1
MODE_ARGS=(-e SPECULATOR=mtp -e MTP=3)
```

```bash
# DFlash2 with its trained default of seven draft tokens.
NAME=jovian-judgement-dflash2-dcp1
MODE_ARGS=(-e SPECULATOR=dflash2 -e NUM_SPECULATIVE_TOKENS=7)
```

Run the selected mode:

```bash
docker rm -f "$NAME" 2>/dev/null || true

docker run -d \
  --name "$NAME" \
  --init \
  --gpus "\"device=${GPU_DEVICES}\"" \
  --network host \
  --ipc host \
  --shm-size 32g \
  -v jovian-judgement-vllm-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e PORT=5001 \
  -e TP=4 \
  -e DCP=1 \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_SEQS=16 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
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

For DFlash2 with DCP4 full-CKV prefill, use the DFlash2 mode above and replace
`-e DCP=1` with:

```bash
  -e DCP=4 \
  -e DCP_CKV_GATHER=1 \
```

## Verify startup

```bash
curl -fsS http://127.0.0.1:5001/health

docker logs "$NAME" 2>&1 | grep -E \
  'speculative_config|B12X PCIe|B12xMxfp8|HUMMING|FlashAttention version 2|split GLM-5.3 cache pages|GPU KV cache size|Graph capturing finished|Application startup complete'

curl -fsS http://127.0.0.1:5001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Reply with exactly READY."}],"temperature":0,"max_tokens":64}'
```

Expected common markers include B12X NVFP4 MoE, B12X PCIe all-reduce, and
512-token target plus recurrent pages. MTP adds `HUMMING` for its MXFP8
experts. DFlash2 adds `B12xMxfp8LinearKernel` and FlashAttention version 2.

## Measured performance

The DCP1 table was measured on physical GPUs 4–7 with stock clocks and PCIe
5.0 x16 links. Every server used the immutable Docker digest above, TP4,
DCP1, FP8 target KV, 512/512 cache pages, B12X PCIe all-reduce, 32 minimum and
maximum NCCL channels, and `MAX_NUM_BATCHED_TOKENS=4096`.

The 32k prefill request contained exactly 32,320 supplied token IDs, used a
unique `cache_salt`, streamed one generated token, and measured client
time-to-first-token (TTFT). One unreported warmup preceded three samples; the
table reports their median. This is an end-to-end TTFT measurement. For
speculative modes it includes the work required to emit the first verified
token, not only isolated target-kernel time.

CC1 decode used `llm-decode-bench` 0.4.29, an empty-context decode cell,
`max_tokens=8192`, greedy sampling, a three-second warmup, and a 30-second
measurement. Three independent cells were measured and the table reports the
median. Accepted length and engine steps per second are server metrics.

| Mode | Speculative configuration | 32k TTFT | 32k prompt throughput | CC1 output | Engine rate | Accepted length |
|---|---|---:|---:|---:|---:|---:|
| No speculative decode | none | 2.0786 s | **15,549 tok/s** | **139.4 tok/s** | — | — |
| MTP:3 | three built-in draft tokens | 2.1360 s | **15,131 tok/s** | **228.0 tok/s** | 90.47 steps/s | 2.52 |
| DFlash2 MXFP8 | seven draft tokens | 2.1157 s | **15,276 tok/s** | **185.5 tok/s** | 74.19 steps/s | 2.49 |

Raw CC1 samples were:

| Mode | Output tok/s samples | Engine steps/s samples | Accepted-length samples |
|---|---|---|---|
| No speculative decode | 139.53, 139.41, 139.38 | — | — |
| MTP:3 | 228.14, 226.01, 228.03 | 90.47, 90.47, 90.48 | 2.52, 2.50, 2.52 |
| DFlash2 MXFP8 | 192.94, 185.49, 178.23 | 74.18, 74.47, 74.19 | 2.60, 2.49, 2.40 |

DFlash2 raw output throughput varies with accepted length. Engine rate remained
within 0.4% across the three DFlash2 samples, so acceptance rather than target
execution speed explains the wider raw tok/s range.

The same image at TP4/DCP4 with DFlash2, seven draft tokens, and full-CKV
gather measured a 2.3803-second median TTFT, or **13,578 prompt tok/s**, for
the same 32,320-token request. DCP1 and DCP4 numbers are separate deployment
contracts and must not be compared as if only one kernel changed.

The CC1 client command was:

```bash
python3 llm_decode_bench.py \
  --host 127.0.0.1 \
  --port 5001 \
  --model GLM-5.3-Flash-NVFP4 \
  --concurrency 1 \
  --contexts 0 \
  --duration 30 \
  --max-tokens 8192 \
  --temperature 0 \
  --decode-warmup-seconds 3 \
  --skip-prefill \
  --no-hw-monitor \
  --show-capacity-limited-values \
  --no-resume \
  --output jovian-judgement-cc1.json
```

## Limitations

- The source pull requests in the source contract are open review units. Use
  the image digest until the required heads are merged and a merged-only image
  is separately qualified.
- Full prefill CUDA-graph capture is unsupported by the GLM GDN backend in this
  source tree. Decode remains graph-captured.
- B12X is the main target backend, but the complete runtime intentionally uses
  PyNCCL outside B12X all-reduce sizes, Humming for MTP experts,
  FlashAttention 2 for DFlash2 local attention, FlashKDA for target prefill,
  and FlashInfer for sampling.
- Raw speculative tok/s is prompt- and acceptance-dependent. Record engine
  steps per second and accepted length with every MTP or DFlash2 comparison.
