# GLM-5.1 NVFP4 + MXFP8 L42-62 Checkpoint

This page documents the reproducible build for the local GLM-5.1 mixed
checkpoint whose sparse expert layers 42-62 are ModelOpt MXFP8 quantized from
the BF16 checkpoint.

The important trap is GLM-5.1 MTP: GLM stores its MTP / next-token-prediction
layer as `model.layers.78.*`, not as `model.mtp.*`. The FP8-PB-WO mixed base
checkpoint has that layer in BF16/unquantized form. The final checkpoint must
therefore transplant `model.layers.78.*` from the NVFP4-MTP checkpoint.

## Output Checkpoint

```text
/root/kld/checkpoints/GLM-5.1-NVFP4-MXFP8-L42-62-BF16src-20260526
```

Build log:

```text
/root/kld/glm51_mxfp8_l42_62_build_gpu8_20260526.log
```

Final state:

| Component | State |
|---|---|
| `model.layers.42-62.mlp.experts.*` | MXFP8 from BF16 weights |
| `model.layers.78.*` | NVFP4 MTP transplanted from `lukealonso/GLM-5.1-NVFP4-MTP` |
| other sparse expert layers | inherited from the base mixed checkpoint |
| dense/shared experts/router/attention/layernorms | inherited from the base mixed checkpoint |

## Source Inputs

Base mixed checkpoint:

```text
/root/.cache/huggingface/hub/models--festr2--glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517/snapshots/e37f1787435d2b2c111a5f5eac924a556a06e257
```

BF16 source checkpoint for the MXFP8 expert weights:

```text
/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
```

NVFP4-MTP source checkpoint for `model.layers.78.*`:

```text
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989
```

## Builder Script

Script:

```text
https://github.com/local-inference-lab/quant-toolkit/blob/codex/glm51-mxfp8-converter-20260526/tools/patch_glm51_experts_mxfp8_from_bf16.py
```

Commit:

```text
https://github.com/local-inference-lab/quant-toolkit/commit/6f406718271f45b1365abf388a00483b66067fd8
```

Branch:

```text
https://github.com/local-inference-lab/quant-toolkit/tree/codex/glm51-mxfp8-converter-20260526
```

The script does three things:

1. Hardlinks unchanged files from the base checkpoint into the output directory.
2. Rewrites the selected sparse expert layer shards as ModelOpt MXFP8 from BF16 tensors.
3. Replaces GLM `model.layers.78.*` with the NVFP4-MTP layer from the MTP source checkpoint.

Do not omit `--mtp-nvfp4-source`. Without it, the output inherits BF16
`model.layers.78.*` from the base mixed checkpoint and is not the production
MTP-capable checkpoint.

## Exact Reproduction Command

```bash
git clone \
  --branch codex/glm51-mxfp8-converter-20260526 \
  https://github.com/local-inference-lab/quant-toolkit.git \
  /root/quant-toolkit-glm51-mxfp8

cd /root/quant-toolkit-glm51-mxfp8

python3 tools/patch_glm51_experts_mxfp8_from_bf16.py \
  --base /root/.cache/huggingface/hub/models--festr2--glm51-nvfp4-w4a16-fp8pbwo-l42-62-20260517/snapshots/e37f1787435d2b2c111a5f5eac924a556a06e257 \
  --bf16-source /root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d \
  --mtp-nvfp4-source /root/.cache/huggingface/hub/models--lukealonso--GLM-5.1-NVFP4-MTP/snapshots/78b7fe365f3905b4e0261a85182fefdbd5137989 \
  --output /root/kld/checkpoints/GLM-5.1-NVFP4-MXFP8-L42-62-BF16src-20260526 \
  --layers 42-62 \
  --gpus 0,1,2,3,4,5,6,7 \
  --force 2>&1 | tee /root/kld/glm51_mxfp8_l42_62_build_gpu8_20260526.log
```

The host needs PyTorch, `safetensors`, and NVIDIA ModelOpt with
`modelopt.torch.quantization.qtensor.MXFP8QTensor`.

## What Gets Converted

The selected tensor pattern is:

```text
model.layers.N.mlp.experts.E.gate_proj.weight
model.layers.N.mlp.experts.E.up_proj.weight
model.layers.N.mlp.experts.E.down_proj.weight
```

For `N = 42..62` and `E = 0..255`, this is:

```text
21 layers * 256 experts * 3 projections = 16128 weight tensors
```

Each selected tensor is loaded from the BF16 checkpoint and quantized with:

```python
MXFP8QTensor.quantize(weight.to(torch.bfloat16).contiguous())
```

The output writes:

```text
*.weight       -> F8_E4M3
*.weight_scale -> U8 MXFP8 scale, group size 32
```

Expected sample shapes:

```text
gate/up weight:       [2048, 6144] F8_E4M3
gate/up weight_scale: [2048, 192]  U8
down weight:          [6144, 2048] F8_E4M3
down weight_scale:    [6144, 64]   U8
```

## MTP Layer 78 Fix

Why this exists:

```text
GLM config has num_nextn_predict_layers=1.
The next-token/MTP layer is model.layers.78.*.
The FP8-PB-WO mixed base stores model.layers.78.* as BF16.
The production MTP checkpoint stores model.layers.78.* as NVFP4.
```

The builder copies the source shards:

```text
model-mtp.safetensors
model-mtp-inputscales.safetensors
```

Then it rewrites `model.safetensors.index.json` so all
`model.layers.78.*` keys point at those NVFP4-MTP shards.

Expected validation for one tensor:

```text
model.layers.78.mlp.experts.0.gate_proj.weight
  file:  model-mtp.safetensors
  shape: [2048, 3072]
  dtype: U8

model.layers.78.mlp.experts.0.gate_proj.input_scale
  file:  model-mtp-inputscales.safetensors
  shape: []
  dtype: F32
```

The output config also marks:

```json
"model.layers.78.mlp.experts": {
  "quant_algo": "NVFP4",
  "group_size": 16
}
```

## Static Validation

Run:

```bash
python3 - <<'PY'
import glob
import json
import os
from safetensors import safe_open

ckpt = "/root/kld/checkpoints/GLM-5.1-NVFP4-MXFP8-L42-62-BF16src-20260526"
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
cfg = json.load(open(os.path.join(ckpt, "config.json")))
wm = idx["weight_map"]
q = cfg["quantization_config"]

print("mxfp8_shards", len(glob.glob(os.path.join(ckpt, "model-mixed-mxfp8-layer*.safetensors"))))
print("fp8pbwo_shards_present", len(glob.glob(os.path.join(ckpt, "model-mixed-fp8pbwo-layer*.safetensors"))))
print("index_fp8pbwo_refs", sum(1 for v in wm.values() if "fp8pbwo" in v))
print("index_mxfp8_refs", sum(1 for v in wm.values() if "mxfp8" in v))
print("layer42", q["quantized_layers"]["model.layers.42.mlp.experts"])
print("layer62", q["quantized_layers"]["model.layers.62.mlp.experts"])
print("layer78", q["quantized_layers"]["model.layers.78.mlp.experts"])

for key in [
    "model.layers.42.mlp.experts.0.gate_proj.weight",
    "model.layers.42.mlp.experts.0.gate_proj.weight_scale",
    "model.layers.78.mlp.experts.0.gate_proj.weight",
    "model.layers.78.mlp.experts.0.gate_proj.input_scale",
]:
    with safe_open(os.path.join(ckpt, wm[key]), framework="pt", device="cpu") as f:
        s = f.get_slice(key)
        print(key, wm[key], s.get_shape(), s.get_dtype())
PY
```

Expected key results:

```text
mxfp8_shards 21
fp8pbwo_shards_present 0
index_fp8pbwo_refs 0
index_mxfp8_refs 32256
layer42 {'quant_algo': 'MXFP8', 'group_size': 32}
layer62 {'quant_algo': 'MXFP8', 'group_size': 32}
layer78 {'quant_algo': 'NVFP4', 'group_size': 16}
model.layers.42.mlp.experts.0.gate_proj.weight model-mixed-mxfp8-layer42.safetensors [2048, 6144] F8_E4M3
model.layers.42.mlp.experts.0.gate_proj.weight_scale model-mixed-mxfp8-layer42.safetensors [2048, 192] U8
model.layers.78.mlp.experts.0.gate_proj.weight model-mtp.safetensors [2048, 3072] U8
model.layers.78.mlp.experts.0.gate_proj.input_scale model-mtp-inputscales.safetensors [] F32
```

## Size

The built checkpoint reports:

```text
523G /root/kld/checkpoints/GLM-5.1-NVFP4-MXFP8-L42-62-BF16src-20260526
```

Each MXFP8 sparse expert layer shard is about:

```text
9,965,863,872 bytes
```

If every remaining NVFP4 sparse expert layer were converted to MXFP8, the
checkpoint would grow by about:

```text
+249.1 GB decimal
+232.0 GiB
```

That assumes the old NVFP4 expert tensors are actually pruned/repacked out of
the old shards. If new MXFP8 shards were only added without pruning dead NVFP4
expert tensors, disk usage would be much larger and should not be treated as the
real model size.
