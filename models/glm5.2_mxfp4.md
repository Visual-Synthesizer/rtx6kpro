# GLM-5.2 FP8 + MXFP4 Experts

This page documents the GLM-5.2 checkpoint with FP8 non-expert weights and
native MXFP4 routed experts on RTX 6000 Pro Blackwell. The serving target is
the local checkpoint below and the public Hugging Face mirror:

```text
/root/models/GLM-5.2-FP8-MXFP4experts
festr2/GLM-5.2-FP8-MXFP4-Experts
```

## Checkpoint Layout

The checkpoint was converted from `zai-org/GLM-5.2-FP8`.

| Tensor family | Storage |
|---|---|
| Dense / attention / MTP tensors | Original FP8 tensors |
| Routed expert weights | MXFP4 e2m1 packed in `uint8` |
| Routed expert scales | UE8M0 K/32 scale grids in `uint8` |
| vLLM quantization marker | `quantization_config.store_dtype = "mxfp4"` |

This layout intentionally avoids the AMD/Quark runtime loader path. vLLM sees
the expert tensors as DeepSeek-style MXFP4 and routes them through B12X
`Mxfp4MoEMethod`.

## Docker Image

```text
voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703
```

| Component | Revision |
|---|---|
| vLLM repo | `https://github.com/local-inference-lab/vllm.git` |
| vLLM branch | `dev/eldritch-enlightenment` |
| vLLM commit | `3f65c5264153b2672e8797f0577da3a1c3ba7198` |
| vLLM build patch | none |
| B12X repo | `https://github.com/lukealonso/b12x.git` |
| B12X commit | `80eb49b7683b32a3a1197c03d69142dd9f835cc7` |
| FlashInfer commit | `5a73a36a7169ec5533ba474bb9204bed765dd297` |
| DeepGEMM commit | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| Build helper | `/root/vllm/blackwell-llm-docker/build-eldritch-enlightenment-mxfp4-glm-cu132.sh` |

Do not update vLLM's Rust `llm-multimodal` dependency when rebuilding this
image. The vLLM branch intentionally pins
`5b558989844d1c7af3e43d0f604069ffd9c06320`, which still exports the old
multimodal API used by the Rust tool/parser crates. Current `llm-multimodal`
main changes that API and breaks clean Rust frontend compilation.

## Launch Helper

Use the checked-in helper:

```bash
cd /root/rtx6kpro

IMAGE=voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703 \
NAME=glm52-mxfp4 \
PORT=8000 \
GPUS=0,1,2,3,4,5,6,7 \
MODE=baseline \
MAX_NUM_SEQS=64 \
scripts/run-glm52-mxfp4-server.sh
```

The helper always unsets `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
`VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` before starting vLLM.

Important defaults:

| Setting | Default |
|---|---|
| TP / DCP | `8 / 1` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| KV cache | `fp8` |
| B12X MoE mode | `B12X_MOE_FORCE_A8=1` |
| Max batched tokens | `8192` |
| Prefix cache | enabled |
| GLM index-cache pattern | full 78-layer pattern |

## DSpark Draft

The baseline image above does not contain the experimental DSpark speculator
implementation. DSpark validation uses the same dependency pins but a vLLM
branch with DSpark enabled:

```text
voipmonitor/vllm:eldritch-enlightenment-dspark-v16140bc-b12x80eb49b-fi5a73a36-cu132-20260703
```

| Component | Revision |
|---|---|
| vLLM branch | `fable/dspark-block-verification-20260703` |
| vLLM commit | `16140bcb5cd8d6375e69d0fcb385a57c2c056d94` |
| B12X / FlashInfer / DeepGEMM | same as baseline image |
| Build helper | `/root/vllm/blackwell-llm-docker/build-eldritch-enlightenment-mxfp4-glm-dspark-cu132.sh` |

The draft checkpoint used for speculative tests is:

```text
RedHatAI/GLM-5.2-speculator.dspark
/root/.cache/huggingface/hub/models--RedHatAI--GLM-5.2-speculator.dspark/snapshots/7985f0391a3d4f309729eb6f79ea086c812f81fb
```

Launch with five draft tokens:

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-dspark-v16140bc-b12x80eb49b-fi5a73a36-cu132-20260703 \
MODE=dspark DSPARK_TOKENS=5 MAX_NUM_SEQS=64 scripts/run-glm52-mxfp4-server.sh
```

Launch with seven draft tokens:

```bash
IMAGE=voipmonitor/vllm:eldritch-enlightenment-dspark-v16140bc-b12x80eb49b-fi5a73a36-cu132-20260703 \
MODE=dspark DSPARK_TOKENS=7 MAX_NUM_SEQS=64 scripts/run-glm52-mxfp4-server.sh
```

For DSpark the helper sets `max_cudagraph_capture_size =
MAX_NUM_SEQS * (DSPARK_TOKENS + 1)` unless `GRAPH` is explicitly overridden.

## Full Sweep

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/glm52-mxfp4-v3f65c52-20260703 \
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703 \
CASES=baseline,dspark5,dspark7 \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
PREFILL_CONTEXTS=8k,64k,128k \
scripts/run-glm52-mxfp4-sweep.sh
```

## Validation Results

Baseline image:
`voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x80eb49b-fi5a73a36-cu132-20260703`

Artifact:
`/root/bench-results/glm52-mxfp4-v3f65c52-baseline-20260703-031608`

Smoke test:

| Test | Result |
|---|---:|
| `/mnt/test.py --port 5660 -L` CJK failures | `0` |
| Short-context generation-only throughput | `~101.5 tok/s` |

Decode sweep, `MAX_NUM_SEQS=64`, MTP/DSpark off:

| Concurrency | Aggregate tok/s | Per-request tok/s |
|---:|---:|---:|
| 1 | 99.7 | 99.7 |
| 16 | 602.7 | 37.7 |
| 32 | 955.1 | 29.8 |
| 64 | 1437.0 | 22.5 |

Coding peak, 5 runs: median `101.0 tok/s`, mean `100.8 tok/s`, max
`101.0 tok/s`, CJK failures `0/5`.

Standalone prefill:

| Prompt | Tokens | TTFT | Client tok/s |
|---:|---:|---:|---:|
| 8k | 8,200 | 1.28 s | 6,412 |
| 64k | 64,514 | 10.52 s | 6,132 |
| 128k | 128,888 | 22.44 s | 5,743 |

DSpark results are pending the DSpark-enabled image build and validation.
