# DeepSeek-V4-Flash-0731 Infernal Invocation r21

**Status: qualified for TP2/DCP1 fixed probabilistic DSpark K5 serving.** The
release artifact is built from pinned public Git revisions and pull-request
heads. Its source locks contain no local source patch or research overlay.
Merging the remaining source pull requests into their canonical branches is
still required before the same tree can be reproduced from branch heads alone.

## TL;DR

Download the committed Compose profile, pull the prebuilt image, and start the
server on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml
docker compose -f docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml pull
docker compose -f docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml up -d
```

The Compose file contains an `image` reference and no `build` section. Its
default profile uses TP2/DCP1, fixed probabilistic DSpark K5, eight admitted
sequences, a 48-row CUDA-graph envelope, a 1,048,576-token per-request limit,
and 4,096-token prefill chunks.

Use the separate target-only profile when speculative decoding is not needed:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-infernal-invocation-cu133-r21.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r21.yml pull
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r21.yml up -d
```

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllmd6cf36a-b12xf6dc512-fi1ac6942-cu133-torch213-20260827-r21` |
| Registry digest | `sha256:ed525dec1a4ac5cf7f19c7cf2fb29661389d71a29ff8de91aade8e6785e10291` |
| Image ID | `sha256:24f19364f0c6a991422bcb436a3e07ab52e66e0eb241aba0b9490e95476a8e3f` |
| Docker source | `local-inference-lab/blackwell-llm-docker@cc2ac998e8f7b5f04d4271a79e6647b4debad3db` |
| Model revision | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@b5f995e73e6b7fe27c9927477e277a151ebcc9e9` |
| vLLM integration tree | `d6cf36ae0dc30d48fd656a3c34a353ec62074922` |
| B12X base | `master@a71c705f1c4710f59129562d26c73e70098e29de` |
| B12X integration tree | `f6dc512eb13ac2c09b2bf53656c704081af64361` |
| LMCache integration tree | `e045d729bc5c4c63a40e13d032f42923de97812f` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, XGrammar 0.2.5 |
| DSpark Compose | [`docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml) |
| Target-only Compose | [`docker-compose-ds4-infernal-invocation-cu133-r21.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-ds4-infernal-invocation-cu133-r21.yml) |

The image labels record every base commit, pull-request head, integration tree,
patch digest, dependency revision, and runtime ABI. The installed Python
packages are compiled artifacts; no source mount or import overlay is used.

## Serving Contract

| Setting | DSpark K5 default | Target-only default |
|---|---:|---:|
| `MODE` | `dspark` | `dspark-mtp0` |
| `BACKEND` | `b12x-a8-dglin` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | `2` / `1` |
| `MAX_NUM_SEQS` | `8` | `32` |
| CUDA graph cap | `48` from `8 * (1 + 5)` | derived by the launcher |
| `MAX_MODEL_LEN` | `1048576` | `1048576` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` |
| `GPU_MEMORY_UTILIZATION` | `0.975` | `0.975` |
| KV format | FP8 compressed MLA | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |

B12X serves sparse attention, routed experts, and tensor-parallel
communication. DGLIN serves FP8 dense projections. `ALLREDUCE_MODE=auto`
delegates graph routing to B12X; it does not run the rejected external
row-by-row startup calibration program.

The DSpark qualification boot reported 1,249,424 aggregate compressed-MLA KV
tokens. One request remains limited by `MAX_MODEL_LEN=1048576`. Native vLLM KV
offload and LMCache are disabled by default because they have independent
ownership contracts:

```bash
# Native vLLM host-memory KV cache, 32 GiB total across TP ranks
KV_OFFLOADING_SIZE=32 \
docker compose -f docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml up -d

# LMCache filesystem-backed mode; do not combine with KV_OFFLOADING_SIZE
LMCACHE_MODE=disk \
docker compose -f docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml up -d
```

The r21 short qualification did not repeat persistent-cache restart testing.
The interfaces and their full restart evidence are specified in the
[r19 serving study](ds4dspark-infernal-invocation-r19.md).

## Source Composition

**Implemented:** the vLLM integration lock contains 31 named pull-request
heads. Seven of those heads are already merged into `dev/infernal-invocation`;
the pinned base predates those merges. The remaining vLLM heads are listed in
[Infernal Invocation DS4 and GLM source merge contract](https://github.com/local-inference-lab/rtx6kpro/issues/67).

**Implemented, review pending:** the B12X integration applies these three pull
requests to `b12x/master` in order:

1. [B12X #243](https://github.com/local-inference-lab/b12x/pull/243) executes
   FP16/BF16 K6/MCG decode with a native fused kernel.
2. [B12X #246](https://github.com/local-inference-lab/b12x/pull/246) uses a
   generation-tagged TP2 peer-push protocol for qualified CUDA-graph shapes.
3. [B12X #247](https://github.com/local-inference-lab/b12x/pull/247) bounds
   native W4A16 route execution by resident tensors.

**Implemented:** all 13 LMCache pull-request heads named by the lock are
already present in
`release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481`.
The LMCache integration patch is empty.

**Research-only and excluded:** the external communication calibration chain,
the runtime FP8 dense-GEMM autotuner, deterministic split-K experiments, and
PDL launch experiments are not present in the image. Their measured variants
did not establish a qualified end-to-end gain over the retained runtime.

## Qualification

The DSpark E2E gate used two RTX PRO 6000 Blackwell GPUs connected through the
same PCIe switch (`PIX`). Both GPUs exposed a 600 W power limit and ran at a
2,692 MHz SM clock during qualification. The server used TP2/DCP1, fixed
probabilistic K5, `MAX_NUM_SEQS=8`, graph cap 48,
`MAX_NUM_BATCHED_TOKENS=4096`, and the default B12X/DGLIN backend.

| Gate | Result |
|---|---|
| Runtime contract | PASS: CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, vLLM package and all dependency versions matched image labels |
| Model load | InstantTensor `BUFFERED` completed |
| CUDA graphs | Target, DSpark, and DFlash context-KV FULL graphs captured |
| Exact response sanity | Returned `r21 validation passed` |
| DSpark C1, 30 seconds | 177.66 aggregate tok/s; 68.45 target steps/s |
| DSpark draft behavior | 2.595 accepted tokens per verifier step; 31.90% strict accepted/drafted |
| Effective KV capacity | 1,249,424 aggregate compressed-MLA tokens |
| TP2 graph peer-push | Exact BF16 payload, signed-zero, transposed storage-dense view, and generation-wrap coverage passed in 9.61 seconds |
| Release composition tests | vLLM, B12X, LMCache tree, patch, and Compose assertions passed |

Emitted-token throughput varies with the generated token trajectory. Target
steps per second is the stable backend metric for DSpark comparisons. The r19
direct-root-port K5 measurement reached 66.1 target steps/s; r21 reached 68.45
steps/s on the PCIe-switch workstation. The hosts differ, so the values show
that the qualified performance regime is retained, not a controlled release
delta.

Only the DSpark C1 E2E gate was repeated for this source composition. Use the
[r19 serving study](ds4dspark-infernal-invocation-r19.md) for TP2/TP4,
concurrency, prefill, one-million-token, and offload measurements. Do not treat
those measurements as r21-specific results.

## Operational Checks

Confirm the expected source and graph contracts after startup:

```bash
docker logs ds4-dspark-infernal-invocation-cu133-r21 2>&1 | \
  grep -E 'version 0.26.1rc0|DS4 launch:|Captured|KV cache size'
```

The launch line must identify `mode=dspark`, `depth=fixed`,
`backend=b12x-a8-dglin`, `tp=2`, `dcp=1`, `max_seqs=8`, and `graph=48`.
Unexpected source tree labels, eager decode dispatch, a different DSpark depth,
or a reused JIT cache from another image invalidates a performance comparison.
