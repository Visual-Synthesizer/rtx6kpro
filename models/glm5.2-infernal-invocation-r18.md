# GLM-5.2 on Infernal Invocation r18

**Status: source-qualified.** This page specifies CUDA 13.3 deployment profiles
for the GLM-5.2 NVFP4 checkpoint and a projection-mixed EXL3 checkpoint. Source,
launcher, sparse-prefill, online K6, routed-expert, and packaging tests passed.
Infernal Invocation r17 remains the physical-GPU qualification reference for
projection-mixed EXL3 TP4/DCP1/MTP3.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18` |
| Registry digest | `sha256:414ec7d0d28358cfd8af0697f330f5c8acbb80e4dc4e5ba69c9fd5b5855ea804` |
| Image ID | `sha256:955e088a85b5378b00275842bc839eea8cb04ca0782ed79eaa3a967d11fd22e5` |
| vLLM integration tree | `f0fa1cefc1865d316c2478525f550e7646addc40` |
| B12X integration tree | `75787c7a7431b3bea414d2ebf5f2b8671b23eb33` |
| Runtime receipt | [`infernal-invocation-r18-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r18-remote-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| TP4 runtime reference | [GLM-5.2 Infernal Invocation r17](glm5.2-infernal-invocation-r17.md) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5, InstantTensor 0.1.9 |

## Projection-Mixed EXL3 TP4

Download and start the four-GPU profile:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-exl3-infernal-invocation-r18.yml
docker compose -f docker-compose-glm52-exl3-infernal-invocation-r18.yml up -d
```

The profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`:

| Setting | Deployment default |
|---|---|
| Parallelism | TP4 / DCP1 / MTP3 |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` |
| CUDA graph cap | 32 rows |
| Context limit | 65,536 tokens |
| Routed experts | Checkpoint-native MCG Trellis K3/K4/K5 per projection |
| Dense and shared projections | Online `trellis-mcg-b6` |
| KV cache | NVFP4 DS-MLA |
| Loading | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` |
| Decode graphs | FULL |

Projection-mixed storage means that every routed-expert gate, up, and down
projection retains its declared K3, K4, or K5 payload. B12X prepares the
checkpoint-native storage through its BTX representation and adopts the
prepared owner without copying or repacking tensors.

The deferred layerwise quantizer takes ownership of InstantTensor buffers that
outlive an iterator step. Eligible BF16 dense and shared projections are
encoded as unpaired MCG Trellis K6 and persisted below `/cache`. Reuse the
release-scoped cache for the same checkpoint revision and quantization
contract.

The checkpoint is rank-sliced for TP4. A TP2 process must reject it before
weight materialization rather than silently reinterpret the shards. The r18
qualification host exposed only GPUs 6 and 7, and the loader produced the
expected error:

```text
rank-sliced EXL3 checkpoint TP does not match runtime: checkpoint=4, runtime=2
```

This fail-closed result validates geometry detection; it is not TP4 runtime
evidence. The r17 runbook records a successful TP4 load, FULL and PIECEWISE
graph capture, coherent output, and 33.42 target steps/s for the same
checkpoint.

## NVFP4 TP8

Download and start the eight-GPU profile:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-nvfp4-infernal-invocation-r18.yml
docker compose -f docker-compose-glm52-nvfp4-infernal-invocation-r18.yml up -d
```

The profile specifies `lukealonso/GLM-5.2-NVFP4`, TP8/DCP1/MTP3, serialized
NVFP4 routed experts, online MXFP8 dense projections, FP8 MLA KV,
InstantTensor `BUFFERED`, and FP8 ring transport. It defaults to
`MAX_NUM_SEQS=32`, graph cap 128, 8,192 batched tokens, and a 262,144-token
model limit.

## Sparse Prefill Contract

GLM sparse attention uses different row semantics during decode and prefill.
Decode kernels consume compact active rows. Prefill kernels consume query rows
whose sparse metadata can span chunked scheduler intervals. The r18 source
keeps those layouts distinct and validates the row count before launching the
sparse kernel. This contract is implemented by
[vLLM #432](https://github.com/local-inference-lab/vllm/pull/432).

The same source composition enforces B12X sparse-MLA geometry, skips absent
fused MTP indexer targets for split EXL projections, and retains online K6
storage across graph capture. These responsibilities are separated so a
checkpoint-format change cannot silently alter sparse-attention row meaning.

## Required Source Pull Requests

The full merge queue is maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).
The GLM-specific source responsibilities are:

| Responsibility | Pull request |
|---|---|
| Fused unpaired MCG K6 dense projection | [B12X #221](https://github.com/local-inference-lab/b12x/pull/221) |
| Projection-mixed K3/K4/K5 routed experts and BTX ownership | [B12X #223](https://github.com/local-inference-lab/b12x/pull/223) |
| Projection-mixed EXL3 loader, online K6 cache, and graph storage | [vLLM #300](https://github.com/local-inference-lab/vllm/pull/300) |
| GLM sparse-MLA runtime contracts | [vLLM #301](https://github.com/local-inference-lab/vllm/pull/301) |
| Split EXL MTP indexer target handling | [vLLM #423](https://github.com/local-inference-lab/vllm/pull/423) |
| Sparse-prefill row semantics | [vLLM #432](https://github.com/local-inference-lab/vllm/pull/432) |

The image contains no private source patch. The official
`vllm-project/vllm/main` branch was audited at
`8f4a7f45c53ab52b17023d3ca804e477daa36a23`; commits outside the Infernal
Invocation source line changed only ROCm CI files and did not supply a GLM
runtime dependency.

## Source Qualification Evidence

| Gate | Result |
|---|---|
| Docker release and source-composition suites | Passed |
| Focused Python tests | 17 passed; `pytest 8.4.1` remains installed in the image |
| Infernal GLM launcher contract | Passed |
| Borrowed InstantTensor ownership | Passed |
| Projection-mixed K3/K4/K5 loading | Passed |
| Online unpaired MCG K6 preparation and cache contract | Passed |
| Sparse-prefill row semantics | Passed |
| EXL rank geometry | TP4 checkpoint correctly rejected TP2 execution |

## Qualification Limits

- **Source-qualified:** projection-mixed EXL3 TP4/DCP1/MTP3 and NVFP4
  TP8/DCP1/MTP3 deployment profiles.
- **Qualified by the r17 runtime receipt:** projection-mixed EXL3
  TP4/DCP1/MTP3 with `MAX_NUM_SEQS=1`, graph cap 4, and a 32,768-token model
  limit.
- **Unsupported by the r18 receipt:** GLM throughput claims, DCP greater than
  one, TP6, larger scheduler capacities, and long-context execution.
- The EXL loader intentionally rejects rank-sliced checkpoints when runtime TP
  does not match checkpoint TP.
- NVFP4 and EXL processes must not share a writable JIT directory.
