# GLM-5.2 Gilded Gnosis r34

Status: **qualified** for the GLM-5.2 R7 mixed-K3/K4/K5 EXL3 profile described
on this page.

Gilded Gnosis r34 serves the GLM-5.2 R7 checkpoint directly from its
mixed-bitrate Trellis expert payloads. Routed experts remain K3, K4, or K5 as
serialized by the checkpoint. BF16 shared experts in layers 3-77 are encoded
into merged K6 gate-up and down projections at model load time and reused from
a content-addressed persistent cache.

The unified image also contains the standard GLM-5.2 NVFP4, NF3 hybrid, DCP,
LMCache, and DeepSeek-V4 serving paths. The r34 qualification receipt covers
only the R7 TP4/DCP1 profile. Other profiles require their own pinned receipts
before performance or quality claims can be transferred to r34.

Historical release data through r33 is retained in
[the Gilded Gnosis release history](glm5.2_v20_history.md).

## Release identity

```text
Image: voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
Registry digest: sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
Local validation image: sha256:0ff4b1de4e950cf48dd0405562908a2f81597f4524698c0291ac2c40514ae17e
Image size: 24,993,088,498 bytes
```

Checkpoint:

```text
Repository: brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78
Revision: 9ab9579774cc432df91567a36f6e9e863e0d4c9f
Shared BF16 shard: model-sharedbf16.safetensors
Shared BF16 shard SHA256: ee1e7d9b2adb5d49c0895dc2f4b7d6d424b108cdb796879eee4c55d040408c6a
Shared BF16 tensor count: 228
```

The checkpoint index records 346,218,639,128 bytes while the referenced shard
files total 346,196,508,248 bytes. The 22,130,880-byte metadata correction is
qualified in
[Hugging Face discussion #3](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78/discussions/3)
at `ed36ad6d767af08845a317b56fe89ce73ed520f4`. That change does not modify
model tensor bytes and is not merged into the checkpoint revision above.

## Start the qualified profile

Docker with NVIDIA Container Toolkit, host IPC, and four Blackwell GPUs with
at least 96 GB each is required. Download the immutable Compose recipe and
start the service:

```bash
curl -fL \
  -o docker-compose-glm52-r7-r34.yml \
  https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/98224d1303c1497eec26c7d92f34a6fa9a58fa82/examples/docker-compose-glm52-r7-v20-r34.yml

docker compose -f docker-compose-glm52-r7-r34.yml pull
docker compose -f docker-compose-glm52-r7-r34.yml up -d
```

The service listens on port 8000. Select another ordered four-GPU set or port
without editing the recipe:

```bash
GPUS=4,5,6,7 PORT=5800 \
  docker compose -f docker-compose-glm52-r7-r34.yml up -d
```

The recipe mounts the Hugging Face cache at `/root/.cache/huggingface` and a
persistent host directory at `/cache`. Preserve `/cache` across container
restarts; it contains EXL3 K6 payloads, CUDA/JIT artifacts, and runtime caches.

## Qualified runtime contract

| Setting | Value | Purpose |
|---|---|---|
| Model family | `glm52-exl3` | Select the EXL3 checkpoint loader and B12X Trellis runtime. |
| TP / DCP | `4 / 1` | Four-way tensor parallelism without decode-context sharding. |
| MTP | `3` | Three speculative draft tokens. |
| MoE | `a16`, backend `b12x` | BF16 activation path for routed experts. |
| Checkpoint quantization | `exl3` | Preserve native K3/K4/K5 routed-expert payloads. |
| Online quantization | `exl3-b6` | Encode eligible BF16 dense and shared projections to K6. |
| MLA KV cache | `nvfp4_ds_mla` | Dynamic NVFP4 compressed-MLA record format. |
| Loader | InstantTensor `BUFFERED`, `INSTANTTENSOR_COPY=0` | Consume borrowed staging buffers without a loader-owned tensor copy. |
| Scheduler | `MAX_NUM_SEQS=8`, `MAX_BATCHED_TOKENS=2048` | Bound concurrent and batched rows used by the qualified graph plan. |
| CUDA graph cap | `32` | Cover target and MTP row counts reachable from eight sequences. |
| Model limit | `65536` | Expose a context limit that fits the qualified MTP3 memory budget. |
| GPU memory utilization | `0.98` | Provide 75,072 KV tokens while preserving successful graph capture. |
| EXL3 prefill capacity | `2048` | Keep the fastest qualified persistent prefill arena. |

`GPU_MEMORY_UTILIZATION=0.97` is insufficient for the MTP3 65,536-token
profile on the validated 96 GB GPUs: KV initialization exposes approximately
47,552 tokens. The qualified `0.98` setting exposes 75,072 tokens.

## Checkpoint and kernel behavior

### Routed experts

The checkpoint metadata contract `r7_routed_experts` identifies K3, K4, or K5
for each routed expert projection. vLLM validates payload geometry and builds
exact gate, up, and down descriptors. B12X dispatches the three MCG codebook
tiers through one mixed-Trellis launch plan for decode and prefill.

Malformed projection counts, descriptor maps, route identifiers, or packed
storage extents fail before execution. The loader does not allocate a silent
BF16 routed-expert fallback.

### Shared experts and dense projections

`ONLINE_QUANT=exl3-b6` encodes eligible BF16 tensors into K6 payloads. Gate and
up projections are merged before encoding; the down projection is encoded
separately. Shared hidden-side rotations are loaded once per layer rather than
expanded for each expert.

The K6 cache key includes checkpoint identity, encoder revision, tensor
geometry, tensor-parallel placement, quantization parameters, and cache schema.
Entries are validated and published atomically. Cache-lock acquisition is
bounded to 600 seconds. A process that cannot acquire a lock performs the
required encoding without publishing into the shared cache.

The default cache directory is `/cache/exl3-online`. Operational controls are:

| Variable | Default | Contract |
|---|---|---|
| `VLLM_EXL3_ONLINE_CACHE_DIR` | `/cache/exl3-online` | Persistent per-rank safetensors directory. |
| `VLLM_EXL3_ONLINE_CACHE_MODE` | `readwrite` | Use `readonly` for a prepopulated cache or `off` for diagnostic encoding without persistence. |
| `VLLM_EXL3_ONLINE_TRELLIS_BITS` | `6` | The R7 online overlay accepts K6 only. |
| `VLLM_EXL3_PREFILL_CAPACITY` | `2048` | Persistent routed-expert prefill arena row capacity. |

Reducing `VLLM_EXL3_PREFILL_CAPACITY` recovers persistent workspace for KV
cache by slicing larger scheduler batches through a smaller arena. A value of
1024 is supported but trades approximately 7-12% prefill throughput in the
measured TP4 profiles. It does not limit prompt length, model context length,
or `MAX_BATCHED_TOKENS`.

### InstantTensor lifetime

InstantTensor `BUFFERED` loading with `INSTANTTENSOR_COPY=0` yields borrowed
staging tensors. Every parameter loader must consume a borrowed tensor before
the iterator advances. The r34 runtime validates this lifetime contract and
does not retain a borrowed staging view in model parameters.

### CUDA graphs

The qualified MTP3 service captures all eight configured FULL decode graph
sizes. Target verification and each of the three MTP draft forwards execute
inside CUDA graphs. Prefill uses vLLM FULL+PIECEWISE capture and does not fall
back to eager execution under the qualified scheduler bounds.

## Performance evidence

Measurements used physical GPUs 4-7 on host `192.168.0.69`. Each GPU is
attached through an independent CPU root port. Every comparison uses the same
host and GPU set; results from the PCIe-switched 16-GPU host are excluded.

Decode cells used context length zero, deterministic sampling, one 3-second
warmup, and a 20-second sustained measurement. Prefill measurements were
uncached.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s | Prefill 8K tok/s | KV tokens |
|---|---:|---:|---:|---:|---:|
| MTP0, GMU 0.97 | 53.80 | 171.30 | 283.05 | 3,253 | 82,816 |
| MTP3, GMU 0.98 | 121.25 | 297.69 | 436.23 | 3,239 | 75,072 |

MTP0 uncached 64K prefill produced 2,966, 3,239, and 3,236 tok/s; the median
was 3,236 tok/s. MTP3 accepted 15,307 of 23,391 draft tokens, a strict draft
acceptance rate of 65.44%.

These numbers describe aggregate server throughput. At C4 and C8, divide the
aggregate value by active users when estimating mean per-user decode rate.

## Quality evidence

Status: **qualified for the tested comparison**.

A paired teacher-forced test compared 2,047 positions over the full vocabulary
against one BF16 teacher. Both candidates used FP8 KV and three repeats.

| Shared-expert representation | Mean KLD | Run SD |
|---|---:|---:|
| Separately serialized K6 | 0.064467 | 0.001973 |
| BF16 source encoded into merged K6 at load time | 0.065339 | 0.001285 |

The absolute mean delta is 0.000872 and the run distributions overlap. The
test did not measure a quality regression from merged runtime K6 encoding.

Limitation: the paired quality test used FP8 KV. The serving recipe uses
NVFP4 DS-MLA KV, so these values do not isolate the cache format's quality
effect.

## Validation gates

- Release composition and immutable patch hashes: pass.
- Installed runtime-contract verifier: pass on GPU 4.
- Focused vLLM suite: 112 passed.
- B12X host suite: 33 passed; 41 GPU tests skipped by the host-only run.
- Installed-image B12X CUDA and CUDA-graph suite: 6 passed.
- MTP0 and MTP3 model startup: pass.
- Deterministic correctness: exact response `r34 validation passed`.
- Request health: 9 successful requests, 0 errors.
- FULL decode and FULL+PIECEWISE prefill graph capture: pass.

The machine-readable receipt includes source hashes, package versions, test
conditions, measurements, conclusions, limitations, and SHA256 checksums for
all raw benchmark artifacts:

https://github.com/local-inference-lab/blackwell-llm-docker/blob/98224d1303c1497eec26c7d92f34a6fa9a58fa82/validation/gilded-gnosis-v20-r34-remote-gpu.json

## Source composition

| Component | Base or package | Composed tree or revision |
|---|---|---|
| vLLM | `dev/gilded-gnosis@e2666d9a65f41fc376607531453cbd57c4c71016` | `4d006a43928cdee01306691a766542c1e9bebb59` |
| B12X | `master@7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b`, package 1.2.1 | `cd3ce190f0f1917402cdfd5773724267cc9a63f8` |
| LMCache | `release/v0.5.2-glm52-dcp-base@9cebd405d0caf4bebe01d694b5a8bf4e3e354314` | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |
| FlashInfer | source revision | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| InstantTensor | source revision | `49b4010afc1cae0441e71fe0b0bffc24fa05e932` |
| Docker launcher | source revision | `7302862b8fcfdc7c06a411a61e1f0fb072258880` |

Runtime packages:

| Package | Version |
|---|---|
| PyTorch | `2.12.0+cu132` |
| CUDA | `13.2.1` |
| cuDNN | `9.22.0.52` |
| cuBLAS | `13.4.1.2` |
| NCCL | `2.30.4` |
| CUTLASS DSL | `4.6.0` |
| XGrammar | `0.2.5` |

R7 source reviews:

- vLLM loader, graph planner, K6 cache, and lifetime integration:
  [vLLM #280](https://github.com/local-inference-lab/vllm/pull/280) at
  `8e7be4d5c97fb86d983bd5f83c825153452efaec`.
- B12X mixed K3/K4/K5 and K6 execution:
  [B12X #144](https://github.com/local-inference-lab/b12x/pull/144) at
  `c8d5f33b4a682a4a0b06e29c816aa24e28313473`.
- Explicit InstantTensor borrowed-buffer mode:
  [vLLM #281](https://github.com/local-inference-lab/vllm/pull/281) at
  `126039af3cc28c667f3b13da3ee0d3abefdd12fe`.

The image contains these exact heads through its archived integration patch.
Their presence in the image does not merge them into vLLM or B12X. The full
ordered merge contract is maintained in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).

## Reproduce the image composition

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 98224d1303c1497eec26c7d92f34a6fa9a58fa82

VLLM_RELEASE_COMPOSITION=reproduce-r34 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The reproduction mode verifies the archived base commits, PR heads, manifest
hashes, integration patch hashes, composed Git trees, runtime symbols, helper
contract, and image labels. Python wheel archives are not bit-reproducible;
source-tree equality and the published registry digest are the immutable
identities for review and deployment.

## Support boundaries

| Profile | Status in r34 |
|---|---|
| GLM-5.2 R7 EXL3, TP4/DCP1, online K6, MTP0/MTP3 | **qualified** under the conditions on this page. |
| GLM-5.2 R7 EXL3 with DCP greater than 1 | **implemented but not qualified by the r34 receipt**. |
| Standard GLM-5.2 NVFP4 and NF3 hybrid profiles | **implemented; use profile-specific qualification records**. |
| R7 context limits above 65,536 tokens | **unsupported by the r34 Compose contract**. |
| Online Trellis bit widths other than K6 | **unsupported by the R7 online overlay**. |

