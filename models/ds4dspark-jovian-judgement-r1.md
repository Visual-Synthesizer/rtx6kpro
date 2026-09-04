# DeepSeek-V4-Flash-0731 Jovian Judgement r1

**Status: qualified for TP2/DCP1 fixed probabilistic DSpark K5 serving.**
This specification covers the source-locked CUDA 13.3 and PyTorch 2.13 image
for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell. The image
supports target-only serving, DSpark K5, and opt-in LMCache. Host KV storage is
disabled unless LMCache is selected explicitly.

## TL;DR

Download the committed DSpark K5 profile, pull the prebuilt image, and start it
on GPUs 0 and 1:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-dspark-jovian-judgement-r1.yml
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r1.yml pull
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r1.yml up -d
```

Use the target-only profile when speculative decoding is not required:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/main/examples/docker-compose-ds4-jovian-judgement-r1.yml
docker compose -f docker-compose-ds4-jovian-judgement-r1.yml pull
docker compose -f docker-compose-ds4-jovian-judgement-r1.yml up -d
```

Both Compose files contain an `image` reference and no `build` section.

## Artifact Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:jovian-judgement-vllm28bc825-b12x8a5b9bf-fi1ac6942-cu133-torch213-20260903-r1` |
| Registry digest | `sha256:9e8cbbe80423510ad77b35fc7414101e2a5d4529ccf6d8c7ee08ab874483bac7` |
| Image ID | `sha256:25bb133dbe37004a7f1b2289c111966357905f6b4c1cbd2bedcc04464c8c0c29` |
| Docker source | `local-inference-lab/blackwell-llm-docker@c95b15f7d7df708745a380f415068c292925c834` |
| Source merge contract | [`rtx6kpro` issue #94](https://github.com/local-inference-lab/rtx6kpro/issues/94) |
| Model revision | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/jovian-judgement@c085b910ebd4a8c89c2c4085cbf17ccaf15a384c` |
| vLLM integration tree | `28bc825a1321bb480fc3294179fd34afeb468389` |
| B12X base | `master@aa90a277a61f9ded46c0f504e37a955b7706659b` |
| B12X integration tree | `8a5b9bfbf59ad61d87efdc8017b91a269d5a319c` |
| LMCache base | `release/v0.5.2-glm52-dcp-base@a128b2e286ebb3556cb43124149e600ff99fe481` |
| LMCache integration tree | `eb4c227f68a4e1c45d6b8edf6b4934e18f6d1f8b` |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, XGrammar 0.2.5, LMCache 0.5.2+glm52dcp.5 |

The image labels record each base revision, pull-request head, integration
tree, generated patch digest, and dependency revision. The runtime uses
compiled installed packages and contains no source overlay.

## Serving Contract

| Setting | DSpark K5 profile | Target-only profile |
|---|---:|---:|
| `MODE` | `dspark` | `dspark-mtp0` |
| `BACKEND` | `b12x-a8-dglin` | `b12x-a8-dglin` |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | `2` / `1` |
| `MAX_NUM_SEQS` | `8` | `32` |
| CUDA graph cap | `48`, derived from `8 * (1 + 5)` | derived from scheduler capacity |
| `MAX_MODEL_LEN` | `1048576` | `1048576` |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | `4096` |
| `GPU_MEMORY_UTILIZATION` | `0.975` | `0.975` |
| KV format | FP8 compressed MLA | FP8 compressed MLA |
| Weight loader | InstantTensor `BUFFERED` | InstantTensor `BUFFERED` |
| Host KV storage | disabled | disabled |

B12X serves compressed sparse attention, routed experts, and qualified
tensor-parallel communication. DGLIN serves the FP8 dense projections. The
launcher captures target, DSpark proposal, and DFlash context-KV decode paths
in CUDA graphs for scheduler-reachable row counts.

## LMCache

LMCache is the supported host-cache integration for this profile. Enable its
RAM tier explicitly:

```bash
LMCACHE_MODE=ram \
LMCACHE_L1_GB=24 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r1.yml up -d
```

Enable RAM plus filesystem storage when cache persistence is required:

```bash
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=256 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
docker compose -f docker-compose-ds4-dspark-jovian-judgement-r1.yml up -d
```

Each serving instance requires a distinct LMCache HTTP port and filesystem
path. The release profiles do not configure native vLLM filesystem offload;
LMCache owns host-side KV movement when either LMCache mode is enabled.

The qualified RAM-tier replay used a 42,090-token request with 41,984
cacheable tokens. Four distinct requests filled 200,352 tokens of a 205,067
token GPU KV pool before replay. The replay restored all 41,984 tokens from
LMCache, read 164 LMCache chunks, completed in 0.313 seconds, and left the
runtime healthy. The uncached store request completed in 3.078 seconds.

## Source Composition

**Implemented, review pending:** the vLLM integration applies two pull
requests to `dev/jovian-judgement`:

1. [vLLM #628](https://github.com/local-inference-lab/vllm/pull/628) registers
   scheduler-reachable speculative row counts before B12X CUDA-graph warmup.
2. [vLLM #630](https://github.com/local-inference-lab/vllm/pull/630) makes
   `ALLREDUCE_MODE=nccl` clear B12X communication overrides before process
   launch.

**Implemented, review pending:** the B12X integration applies two pull
requests to `b12x/master`:

1. [B12X #246](https://github.com/local-inference-lab/b12x/pull/246) makes
   the TP2 CUDA-graph peer-push protocol generation-safe.
2. [B12X #302](https://github.com/local-inference-lab/b12x/pull/302) provides a
   valid W4A8 routed-expert profiling oracle.

**Implemented, review pending:**
[LMCache #44](https://github.com/local-inference-lab/LMCache/pull/44) handles
interleaved 64-head KV pages and transfers blocks with their physical
dimension-zero stride. Thirteen earlier LMCache integration pull requests are
already present in the pinned base branch.

**Unsupported:** native vLLM filesystem L2 is not part of the release serving
contract.

## Qualification Evidence

Two RTX PRO 6000 Blackwell GPUs connected through one PCIe switch ran TP2/DCP1
with the exact vLLM, B12X, and LMCache trees recorded above. The final registry
artifact differs from the GPU-qualified local artifact only by OCI metadata;
its filesystem layers are identical.

| Gate | Conditions and result |
|---|---|
| Source composition | Static release assertions passed for vLLM, B12X, LMCache, Compose defaults, and package versions |
| Model loading | InstantTensor `BUFFERED` completed |
| CUDA graphs | Target, fixed K5 DSpark proposal, and DFlash context-KV capture completed |
| Deterministic generation | Returned the requested exact answer `42` |
| Strict tool call | `strict: true` with `tool_choice=required` returned the requested calculator call with `a=19`, `b=23` |
| DSpark C1 | 211.53 aggregate tok/s over 30 seconds; 78.4 target steps/s; 2.70 accepted output tokens per target step |
| LMCache RAM replay | 41,984/41,984 cacheable prompt tokens restored; 164 chunks read; no runtime error |
| Registry publication | Docker Hub digest matches the value in Artifact Identity |

Emitted-token throughput depends on DSpark acceptance for the generated token
trajectory. Target steps per second is the acceptance-normalized backend
metric. The C1 result is a single-workload throughput qualification, not a
cross-release performance claim.

## Qualification Limits

- **Qualified:** TP2/DCP1 fixed probabilistic DSpark K5, B12X W4A8, FP8
  compressed MLA KV, target/draft/context-KV CUDA graphs, strict required tool
  calls, and LMCache RAM store/evict/replay.
- **Implemented:** target-only mode and LMCache filesystem storage.
- **Unsupported by this receipt:** TP other than two, DCP greater than one,
  K7 performance, a full 1,048,576-token request, LMCache filesystem restart,
  and model-quality evaluation.
- Performance was measured on a PCIe-switch workstation and is not directly
  comparable with direct-root-port measurements.
- Keep the release-scoped `/cache` mount. The first request for an uncovered
  B12X shape may compile kernels.
