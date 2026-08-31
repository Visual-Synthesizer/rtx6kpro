# GLM-5.1 MXFP4 Hybrid Native Checkpoint

This page documents the reproducible GLM-5.1 MXFP4 hybrid checkpoint and the
small vLLM overlay image used to load it without Quark runtime hooks.

Status: built on 2026-06-10. The tensor checkpoint was prepared locally and the
vLLM changes live on a separate local-inference-lab branch. Use the native
checkpoint path below with the Docker Compose recipe on this page; do not mount
the old Quark/index-filter runtime overlays.

## TL;DR: Run The Clean Native Checkpoint

Use this checkpoint:

```text
https://huggingface.co/festr2/GLM-5.1-MXFP4-Hybrid-Native-20260610
```

Use this image:

```text
voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610
```

Create `/root/glm51-mxfp4/compose.yaml` from the **Docker Compose** section
below, then start the default TP8/DCP1/no-MTP server:

```bash
mkdir -p /root/glm51-mxfp4
cd /root/glm51-mxfp4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=5329 \
TP_SIZE=8 \
DCP_SIZE=1 \
MTP=0 \
docker compose up -d
```

Watch startup and verify the expected native MXFP4 + FlashInfer Cutlass path:

```bash
docker logs -f glm51-native-mxfp4

docker logs glm51-native-mxfp4 2>&1 | \
  rg "FLASHINFER_CUTLASS_MXFP4_MXFP8|standard SwiGLU|B12X_MLA_SPARSE|V2 Model Runner"
```

Smoke test:

```bash
python3 /mnt/test.py --port 5329 -L
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 \
  --concurrency 1 \
  --contexts 0k \
  --max-tokens 512 \
  --skip-prefill
```

To enable MTP, restart with:

```bash
cd /root/glm51-mxfp4
MTP=1 NUM_SPECULATIVE_TOKENS=3 DRAFT_SAMPLE_METHOD=probabilistic docker compose up -d
```

Important runtime constraints:

```text
Do not set B12X_MOE_FORCE_A16 for this checkpoint.
Do not mount old Quark/index-filter/load-diagnostic overlays.
The compose entrypoint intentionally unsets NCCL_GRAPH_FILE and NCCL_GRAPH_DUMP_FILE.
```

## Final Artifacts

Hugging Face checkpoint:

```text
https://huggingface.co/festr2/GLM-5.1-MXFP4-Hybrid-Native-20260610
```

Local build/source checkpoint path:

```text
/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610
```

Docker image:

```text
voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610
```

Docker image digest:

```text
sha256:2bcffff1b4932854a8b6a8fcc1a1cdb49e2d83249ddc394cb659e555b9f3b46d
```

vLLM branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/glm51-native-mxfp4-20260610
```

vLLM commit:

```text
f07a09da7aa1792f1383c0653480fbd9d836f6eb
```

Base image:

```text
voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608
```

Base image digest:

```text
sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

## Source Inputs

Luke NVFP4-MTP source:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

AMD MXFP4 source:

```text
/root/.cache/huggingface/hub/models--amd--GLM-5.1-MXFP4/snapshots/4dded8b53961222a5d98378fa8b975bc8816599d
```

Intermediate Quark hybrid:

```text
/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-QuarkExcludeFix-20260609
```

Clean repack:

```text
/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610
```

Clean repack log:

```text
/root/kld/repack_glm51_hybrid_mxfp4_clean_20260610.log
```

## Tensor Lineage

The checkpoint is not a pure AMD checkpoint and not a metadata rename of Luke's
NVFP4 checkpoint.

| Tensor group | Source |
|---|---|
| Routed expert `mlp.experts.*.{gate,up,down}_proj.weight` | AMD `GLM-5.1-MXFP4` |
| Routed expert `mlp.experts.*.{gate,up,down}_proj.weight_scale` | AMD `GLM-5.1-MXFP4` |
| Attention, routers, dense layers, shared experts, norms, embeddings, LM head | Luke `GLM-5.1-NVFP4-MTP` |
| GLM MTP / next-token layer `model.layers.78.*` | Luke `GLM-5.1-NVFP4-MTP` |

AMD MXFP4 routed experts have `weight` plus `weight_scale`. The scale is uint8
E8M0 with group size 32. Luke ModelOpt NVFP4 tensors are structurally different:
they include `weight_scale_2` and `input_scale`. Therefore this checkpoint is
handled as native MXFP4 routed MoE, not converted into Luke/ModelOpt NVFP4.

## Clean Repack

The Quark hybrid initially referenced original source shard names. The clean
repack makes a standalone checkpoint whose index only points to local
`model-xxxxx-of-00365.safetensors` shards. This removes the old exact-index
runtime loader overlay.

```bash
python3 /root/kld/tools/repack_safetensors_from_index.py \
  --src /root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-QuarkExcludeFix-20260609 \
  --dst /root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610 \
  --force 2>&1 | tee /root/kld/repack_glm51_hybrid_mxfp4_clean_20260610.log
```

Expected clean-repack state:

```text
index entries: 118238
unique shard files: 365
bad index refs outside model-*: 0
```

Validate the index:

```bash
python3 - <<'PY'
import json
import os

ckpt = "/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610"
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
refs = list(idx["weight_map"].values())
print("index_entries", len(refs))
print("unique_files", len(set(refs)))
print("bad_refs", sum(1 for name in refs if not name.startswith("model-")))
PY
```

## Native MXFP4 Config

The final native checkpoint is a hardlink clone of the clean repack. Tensor data
is not duplicated. Only `config.json` is replaced so vLLM selects the native
`mxfp4` quantization path instead of Quark.

```bash
SRC=/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610
DST=/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610

cp -al "$SRC" "$DST"

python3 - <<'PY'
import json
import os
import tempfile

dst = "/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610"
path = os.path.join(dst, "config.json")
with open(path) as f:
    cfg = json.load(f)

cfg["quantization_config"] = {
    "quant_method": "mxfp4",
    "format": "native_glm_routed_moe",
    "weight_dtype": "mxfp4",
    "weight_scale_format": "e8m0",
    "weight_group_size": 32,
    "activation_dtype": "mxfp8_dynamic",
    "codex_notes": {
        "source": "Hardlinked from clean Luke NVFP4-MTP + AMD MXFP4 routed-expert repack.",
        "why": "Avoid Quark runtime config; route GLM routed experts through native vLLM MXFP4 MoE.",
        "launch": "Use --moe-backend flashinfer_cutlass_afp8 for FlashInfer Cutlass MXFP4xMXFP8.",
        "previous_quant_method": "quark",
    },
}

fd, tmp = tempfile.mkstemp(dir=dst, prefix=".config.", suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(cfg, f, indent=2, sort_keys=False)
    f.write("\n")
os.replace(tmp, path)
PY
```

Validate that only config metadata was separated:

```bash
python3 - <<'PY'
import json
import os

src = "/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610"
dst = "/root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610"

for rel in ["config.json", "model-00001-of-00365.safetensors", "model.safetensors.index.json"]:
    ss = os.stat(os.path.join(src, rel))
    ds = os.stat(os.path.join(dst, rel))
    print(rel, "same_inode", (ss.st_dev, ss.st_ino) == (ds.st_dev, ds.st_ino))

for root in [src, dst]:
    cfg = json.load(open(os.path.join(root, "config.json")))
    print(os.path.basename(root), cfg["quantization_config"]["quant_method"])
PY
```

Expected output:

```text
config.json same_inode False
model-00001-of-00365.safetensors same_inode True
model.safetensors.index.json same_inode True
GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-CleanRepack-20260610 quark
GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610 mxfp4
```

## vLLM Changes

Branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/glm51-native-mxfp4-20260610
```

Changed files:

| File | Purpose |
|---|---|
| `vllm/model_executor/layers/quantization/mxfp4.py` | For non-`gpt_oss` models, route `Mxfp4Config` to the native `Mxfp4MoEMethod`. |
| `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | Add native FlashInfer Cutlass MXFP4 conversion for GLM contiguous `[gate, up]` layout, zero MoE biases when absent, and interleaved MXFP8 scales. |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` | Apply GPT-OSS alpha/beta/clamp SwiGLU defaults only to `model_type=gpt_oss`; GLM uses standard SwiGLU. |

Static validation:

```bash
cd /root/vllm/worktrees/vllm-glm51-native-mxfp4-20260610
python3 -m py_compile \
  vllm/model_executor/layers/quantization/mxfp4.py \
  vllm/model_executor/layers/fused_moe/oracle/mxfp4.py \
  vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
git diff --check HEAD~
```

## Overlay Docker Image

The image is an overlay on top of the existing Black Benediction image. It does
not rebuild vLLM, B12X, FlashInfer, DeepGEMM, CUDA, or Torch. It only replaces
the three Python files above.

Dockerfile:

```Dockerfile
FROM voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608

LABEL org.opencontainers.image.title="GLM-5.1 native MXFP4 vLLM overlay"
LABEL org.opencontainers.image.source="https://github.com/local-inference-lab/vllm/tree/codex/glm51-native-mxfp4-20260610"
LABEL org.opencontainers.image.revision="f07a09da7aa1792f1383c0653480fbd9d836f6eb"
LABEL org.opencontainers.image.base.name="voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608"

COPY vllm/model_executor/layers/quantization/mxfp4.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/mxfp4.py
COPY vllm/model_executor/layers/fused_moe/oracle/mxfp4.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
COPY vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
```

Build and push:

```bash
docker build \
  -f /root/vllm/build/glm51-native-mxfp4-f07a09d/Dockerfile \
  -t voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610 \
  /root/vllm/worktrees/vllm-glm51-native-mxfp4-20260610

docker push voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610
```

Offline image sanity:

```bash
docker run --rm --network none --entrypoint /bin/bash \
  -v /root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610:/model:ro \
  voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610 \
  -lc '/opt/venv/bin/python - <<PY
from transformers import AutoConfig
from vllm.model_executor.layers.quantization import get_quantization_config
cfg = AutoConfig.from_pretrained("/model", trust_remote_code=True)
qc = cfg.quantization_config
cls = get_quantization_config("mxfp4")
print("model_type", cfg.model_type)
print("quant_method", qc.get("quant_method"))
print("format", qc.get("format"))
print("quant_cls", cls.__name__)
print("quant_name", cls.get_name())
PY'
```

Expected output:

```text
model_type glm_moe_dsa
quant_method mxfp4
format native_glm_routed_moe
quant_cls Mxfp4Config
quant_name mxfp4
```

## Runtime Rules

Use `--moe-backend flashinfer_cutlass_afp8`. This selects the native
FlashInfer Cutlass MXFP4xMXFP8 MoE path.

Do not set `B12X_MOE_FORCE_A16` for this checkpoint. That flag only applies to
B12X MoE W4A16 decode. This checkpoint uses FlashInfer Cutlass MoE.

Do not mount the old runtime overlays for this native checkpoint:

```text
/root/vllm/runtime-overlays/glm51-mxfp4-quark-mxfp8
/root/vllm/runtime-overlays/glm51-mxfp4-index-filter
/root/vllm/runtime-overlays/glm51-mxfp4-load-diagnostic
```

Always unset empty NCCL graph variables before launching:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
```

## Docker Compose

This compose recipe is the intended native launch. Defaults use TP8, DCP1, port
`5329`, KV fp8, 64 max sequences, and MTP disabled. Enable MTP with `MTP=1`.

```yaml
services:
  glm51-mxfp4:
    image: ${IMAGE:-voipmonitor/vllm:glm51-native-mxfp4-f07a09d-cu132-20260610}
    container_name: ${NAME:-glm51-native-mxfp4}
    network_mode: host
    ipc: host
    shm_size: 32g
    runtime: nvidia
    gpus: all
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - /mnt:/mnt
      - /cache:/cache
      - /root/.cache/huggingface:/root/.cache/huggingface
      - /root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610:/model:ro
    environment:
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      CUDA_DEVICE_MAX_CONNECTIONS: "32"
      OMP_NUM_THREADS: "16"
      CUTE_DSL_ARCH: sm_120a
      NCCL_IB_DISABLE: "1"
      NCCL_P2P_LEVEL: SYS
      NCCL_PROTO: LL,LL128,Simple
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SAFETENSORS_FAST_GPU: "1"
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      VLLM_USE_AOT_COMPILE: "1"
      VLLM_USE_BREAKABLE_CUDAGRAPH: "0"
      VLLM_USE_MEGA_AOT_ARTIFACT: "1"
      VLLM_USE_FLASHINFER_SAMPLER: "1"
      VLLM_USE_B12X_FP8_GEMM: "1"
      VLLM_USE_B12X_SPARSE_INDEXER: "1"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_ENABLE_PCIE_ALLREDUCE: "1"
      VLLM_PCIE_ALLREDUCE_BACKEND: b12x
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 64KB
      USES_B12X: "True"
      B12X_DENSE_SPLITK_TURBO: "1"
      MODEL: /model
      MTP_MODEL: /model
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.1-MXFP4-NATIVE}
      PORT: ${PORT:-5329}
      TP_SIZE: ${TP_SIZE:-8}
      DCP_SIZE: ${DCP_SIZE:-1}
      MTP: ${MTP:-0}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.94}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      MAX_NUM_BATCHED_TOKENS: ${MAX_NUM_BATCHED_TOKENS:-8192}
      MAX_CUDAGRAPH_CAPTURE_SIZE: ${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}
      NUM_SPECULATIVE_TOKENS: ${NUM_SPECULATIVE_TOKENS:-3}
      DRAFT_SAMPLE_METHOD: ${DRAFT_SAMPLE_METHOD:-probabilistic}
      HF_OVERRIDES: '{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
    entrypoint:
      - bash
      - -lc
      - |
        set -euo pipefail
        unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
        SPEC_ARGS=()
        if [ "$${MTP}" = "1" ]; then
          SPEC_CONFIG=$$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"flashinfer_cutlass_afp8","draft_sample_method":"%s"}' "$${MTP_MODEL}" "$${NUM_SPECULATIVE_TOKENS}" "$${DRAFT_SAMPLE_METHOD}")
          SPEC_ARGS=(--speculative-config "$${SPEC_CONFIG}")
        fi
        cd /
        exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$${MODEL}" \
          --served-model-name "$${SERVED_MODEL_NAME}" \
          --trust-remote-code \
          --host 0.0.0.0 \
          --port "$${PORT}" \
          --tensor-parallel-size "$${TP_SIZE}" \
          --pipeline-parallel-size 1 \
          --decode-context-parallel-size "$${DCP_SIZE}" \
          --dcp-comm-backend ag_rs \
          --dcp-kv-cache-interleave-size 1 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --load-format fastsafetensors \
          --async-scheduling \
          --gpu-memory-utilization "$${GPU_MEMORY_UTILIZATION}" \
          --max-num-batched-tokens "$${MAX_NUM_BATCHED_TOKENS}" \
          --max-num-seqs "$${MAX_NUM_SEQS}" \
          --max-cudagraph-capture-size "$${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
          --kv-cache-dtype fp8 \
          --block-size 128 \
          --attention-backend B12X_MLA_SPARSE \
          --linear-backend b12x \
          --moe-backend flashinfer_cutlass_afp8 \
          --hf-overrides "$${HF_OVERRIDES}" \
          -cc.pass_config.fuse_allreduce_rms=True \
          "$${SPEC_ARGS[@]}"
```

## Runtime Validation Checklist

After starting the native image, confirm the logs contain the expected path:

```bash
docker logs glm51-native-mxfp4 2>&1 | rg "Mxfp4|MXFP4|FlashInfer|SwiGLU|model_type|B12X_MLA"
```

The important signals are:

```text
model_type=glm_moe_dsa
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend
Using GLM contiguous w13 layout
Using standard SwiGLU parameters
Using V2 Model Runner
Using AttentionBackendEnum.B12X_MLA_SPARSE backend
```

Basic smoke:

```bash
python3 /mnt/test.py --port 5329 -L
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port 5329 \
  --concurrency 1 \
  --contexts 0k \
  --max-tokens 512 \
  --skip-prefill
```

## Previous Runtime State

The older working runtime used Quark metadata plus overlays for activation
override, index filtering, GLM w13 layout, zero MoE biases, and standard GLM
SwiGLU. That was useful for proving the tensor path, but it should not be the
long-term reproduction target.

The native branch/image folds the GLM w13 layout and SwiGLU fixes into vLLM and
uses a clean standalone checkpoint. It should not need Quark runtime config or
the exact-index loader overlay.

## Lucifer FlashInfer Cutlass Runtime

Status: validated on 2026-06-10 as a Lucifer overlay smoke run. This is not the
same attention path as the B12X GLM runtime above. Lucifer's DS4 sparse MLA
backend currently rejects GLM's sparse/indexer shape, so the working GLM path
uses a dense MLA fallback plus FlashInfer Cutlass MXFP4xMXFP8 MoE.

Lucifer image:

```text
voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610
```

Lucifer image digest:

```text
sha256:3b6806ca1a38352cb0b0549cc23ae8a88c5d54f6c0042080d0b35f647b44eafb
```

Lucifer vLLM branch:

```text
https://github.com/local-inference-lab/vllm/tree/codex/lucifer-glm51-mxfp4-20260610
```

Lucifer vLLM commit:

```text
bbbb9fc3b32f99c1fa6f17cf5c497d3f89a86682
```

Base Lucifer image:

```text
voipmonitor/vllm:lucifer
voipmonitor/vllm:lucifer-vllm7c6bbf4-fi3395b41aa8d-dg324aced12c-tk9801a7-cu132-20260609
sha256:76f5f2cb4942d5b175908192ac07be81df077fe28cd5d3f8c7c92611895e14d4
```

Additional Lucifer changes compared with the native B12X overlay:

| File | Purpose |
|---|---|
| `vllm/model_executor/models/deepseek_v2.py` | Add env-gated `VLLM_GLM_FORCE_DENSE_MLA=1` fallback so GLM can bypass Lucifer sparse MLA/indexer modules; skip sparse indexer checkpoint tensors in this dense mode. |
| `vllm/v1/attention/ops/triton_decode_attention.py` | Use `num_stages=1` for dense GLM MLA decode with effective QK dim 576 to stay under Blackwell shared-memory limits during CUDA graph capture. |

Why this is necessary:

```text
SPARSE_MLA_SM120 is DS4-oriented in Lucifer and rejects GLM head_size=576 /
GLM sparse-indexer assumptions. Without the dense fallback the selector cannot
find a valid attention backend. With dense fallback, TRITON_MLA works, but needs
the num_stages=1 shared-memory fix for CUDA graph capture.
```

Overlay Dockerfile:

```Dockerfile
FROM voipmonitor/vllm:lucifer

LABEL org.opencontainers.image.title="GLM-5.1 native MXFP4 Lucifer dense-MLA overlay"
LABEL org.opencontainers.image.source="https://github.com/local-inference-lab/vllm/tree/codex/lucifer-glm51-mxfp4-20260610"
LABEL org.opencontainers.image.revision="bbbb9fc3b32f99c1fa6f17cf5c497d3f89a86682"
LABEL org.opencontainers.image.base.name="voipmonitor/vllm:lucifer"

COPY vllm/model_executor/models/deepseek_v2.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py
COPY vllm/model_executor/layers/quantization/mxfp4.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/mxfp4.py
COPY vllm/model_executor/layers/fused_moe/oracle/mxfp4.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
COPY vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
COPY vllm/v1/attention/ops/triton_decode_attention.py /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_decode_attention.py
```

Build and push:

```bash
docker build \
  -f /root/vllm/build/lucifer-glm51-mxfp4-densemla-bbbb9fc/Dockerfile \
  -t voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610 \
  /root/vllm/worktrees/vllm-lucifer-glm51-mxfp4-20260610

docker push voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610
```

Validated smoke launch. This was intentionally small-context validation
(`--max-model-len 4096`, `--max-num-seqs 8`) to prove the Lucifer
FlashInfer-Cutlass MoE path and CUDA graph capture.

```bash
docker run -d --name glm51-lucifer-mxfp4 \
  --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt:/mnt \
  -v /cache:/cache \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/kld/checkpoints/GLM-5.1-LukeNVFP4-MTP-AMD-MXFP4-Routed-W4A4AsMXFP8-NativeMXFP4-20260610:/model:ro \
  -e CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_GLM_FORCE_DENSE_MLA=1 \
  voipmonitor/vllm:lucifer-glm51-mxfp4-densemla-bbbb9fc-cu132-20260610 \
  /bin/bash -lc 'set -euo pipefail; unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_ENABLE_PCIE_ALLREDUCE VLLM_PCIE_ALLREDUCE_BACKEND VLLM_CPP_AR_1STAGE_NCCL_CUTOFF VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS VLLM_RTX6K_FUSED_ALLREDUCE_ADD VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER VLLM_CACHE_DIR; exec vllm serve /model \
    --served-model-name GLM-5.1-MXFP4-NATIVE-LUCIFER \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 5331 \
    --load-format fastsafetensors \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 64 \
    --compilation-config="{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]}" \
    --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --enable-chunked-prefill \
    --enable-flashinfer-autotune \
    --enable-prefix-caching \
    --kernel-config.moe_backend flashinfer_cutlass'
```

Expected log signals:

```text
Forcing dense MLA for GLM because VLLM_GLM_FORCE_DENSE_MLA=1.
Using TRITON_MLA attention backend
Using FLASH_ATTN MLA prefill backend
Using native MXFP4 MoE method for model_type=glm_moe_dsa.
Using 'FLASHINFER_CUTLASS_MXFP4_MXFP8' Mxfp4 MoE backend.
Using contiguous w13 layout for FlashInfer Cutlass MXFP4 MoE.
Using standard SwiGLU parameters for FlashInfer MXFP4 MoE (model_type=glm_moe_dsa).
Graph capturing finished
Starting vLLM server on http://0.0.0.0:5331
```

Validated smoke result:

| Test | Result |
|---|---|
| `/v1/models` | `GLM-5.1-MXFP4-NATIVE-LUCIFER`, `max_model_len=4096` |
| KV cache budget | `746,496` tokens, `2916` blocks × `256` |
| `python3 /mnt/test.py --port 5331 -L` | first iteration completed, `1892` completion tokens, `0` CJK characters |
| Smoke generation-only speed | `64.14 tok/s` |
| `llm_decode_bench.py --port 5331 --concurrency 1 --contexts 0k --max-tokens 256 --skip-prefill` | `65.2 tok/s`, TTFT/ITL `84/15 ms` |

Open item: this proves Lucifer compatibility for the checkpoint and
FlashInfer-Cutlass MXFP4 MoE, but it is still a dense attention fallback. A
proper performance path would need a GLM-compatible Lucifer sparse MLA backend
instead of forcing dense MLA.
