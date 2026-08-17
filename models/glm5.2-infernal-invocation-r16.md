# GLM-5.2 on Infernal Invocation r16

**Status: source-qualified.** The image contains the GLM-5.2 NVFP4 TP8 and
projection-mixed EXL3 TP4 serving profiles. Focused release-image tests qualify
the loader, mixed-rate expert schema, online K6 cache, and B12X sparse-MLA
contracts. Full-checkpoint E2E execution on this exact image requires community
hosts with the corresponding GPU capacity.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm5beffc4-b12xa4a0bc8-fi1ac6942-cu133-torch213-20260817-r16` |
| Registry digest | `sha256:ff9d4f2402ed88b1ae7ca3a6886c80a64d72993f1a593380c8cb6f193437567d` |
| vLLM integration tree | `5beffc48f7cd9d4ade076e4b6d1f117ac8e79d4a` |
| B12X integration tree | `a4a0bc8a8f5e56dbef85f9b46b0d74f6e8edb491` |
| Runtime receipt | [`infernal-invocation-r16-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r16-remote-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

SQG checkpoint support is unsupported and absent from the r16 source
composition.

## NVFP4 TP8 Profile

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-nvfp4-infernal-invocation-r16.yml
docker compose -f docker-compose-glm52-nvfp4-infernal-invocation-r16.yml up -d
```

The profile serves `lukealonso/GLM-5.2-NVFP4` with this contract:

| Setting | Value |
|---|---|
| Parallelism | TP8 / DCP1 / MTP3 |
| Scheduler | `MAX_NUM_SEQS=32`, `MAX_BATCHED_TOKENS=8192` |
| CUDA graph cap | 128 rows |
| Context limit | 262,144 tokens |
| Routed experts | Serialized NVFP4 with B12X W4A16 execution |
| Dense projections | Online MXFP8 without ignored projections |
| KV cache | FP8 MLA |
| Loading | InstantTensor `BUFFERED` |
| FP8 transport | `F8_DMA=ring` |

## Projection-Mixed EXL3 TP4 Profile

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-glm52-exl3-infernal-invocation-r16.yml
docker compose -f docker-compose-glm52-exl3-infernal-invocation-r16.yml up -d
```

The profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78` revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`:

| Setting | Value |
|---|---|
| Parallelism | TP4 / DCP1 / MTP3 |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` |
| CUDA graph cap | 32 rows |
| Context limit | 65,536 tokens |
| Routed experts | Checkpoint-native Trellis/MCG K3/K4/K5 per projection |
| Dense and shared projections | Online `trellis-mcg-b6` |
| KV cache | NVFP4 DS-MLA |
| Loading | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` |
| Decode graphs | FULL |

The first load encodes eligible BF16 tensors as K6 and stores the derived
payload below the release-scoped `/cache` mount. The cache key includes the
checkpoint revision and quantization contract. Reuse the same persistent cache
to avoid repeating the encoding work.

Projection-mixed means that each routed projection retains its own K3, K4, or
K5 payload. The loader does not assign one bitrate to the complete layer.
B12X dispatches each projection to the matching native CuTe DSL kernel and
retains graph-stable output and workspace addresses.

## Source Validation

| Gate | Result |
|---|---|
| B12X focused suite | 309 passed and 18 skipped in the release source tree |
| Projection-mixed loading and online K6 | Loader, cache-key, payload, and packaging contracts passed |
| Routed-expert execution | Mixed K3/K4/K5, inactive-route, launch-policy, and graph replay contracts passed |
| Sparse MLA | Cache dtype, metadata, sparse-request coverage, and workspace contracts passed |
| Docker release suites | DeepSeek and GLM Compose and source-lock contracts passed |

The sparse-MLA tests cover cache-dtype rejection, NVFP4 DS-MLA acceptance,
nested dense-decode metadata, complete sparse-request coverage, and shared
capture/runtime workspace state.

## Community E2E Gate

An E2E report must include:

- image tag and registry digest;
- checkpoint repository and revision;
- TP, DCP, MTP, scheduler, graph, and KV-cache settings;
- physical GPU order and PCIe topology;
- startup lines that confirm FULL graph capture and the selected B12X kernels;
- one deterministic arithmetic request and one warmed CC1 measurement;
- engine errors and the allocated KV-token count.

The Infernal Invocation r11 receipt measured 182.54 aggregate tok/s for NVFP4
TP8/DCP1/MTP3 and 127.22 to 131.45 aggregate tok/s for projection-mixed EXL3
TP4/DCP1/MTP3. Those exact-image measurements are regression references, not
r16 results.

## Qualification Limits

- **Source-qualified:** projection-mixed K3/K4/K5 loading, cached online K6,
  GLM sparse-MLA runtime contracts, launch profiles, and CUDA graph tests.
- **Implemented:** NVFP4 TP8/DCP1/MTP3 and projection-mixed EXL3
  TP4/DCP1/MTP3 launch paths.
- **Unqualified on the exact image:** full-checkpoint startup, CC1 throughput,
  prefill throughput, DCP greater than one, TP6, and long-context quality.
- NVFP4 and EXL3 processes must not share a writable JIT directory.
