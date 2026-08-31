"""Convert GLM-5.2-FP8 -> DS4-Flash-style mixed checkpoint:
FP8 dense/attention (passthrough) + MXFP4 routed experts (offline requant).

Expert tensors `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`
(fp8e4m3 + fp32 `weight_scale_inv` per 128x128 block) become:
  .weight        uint8  [rows, k/2]   packed e2m1, even k in the LOW nibble
  .weight_scale  uint8  [rows, k/32]  e8m0 biased exponent (OCP MX recipe)
`weight_scale_inv` is dropped for experts. Everything else passes through.
"""
import json
import math
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1]
DST = sys.argv[2]
os.makedirs(DST, exist_ok=True)
DEV = "cuda:0"

EXP_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)

# e2m1 magnitude grid and midpoints for nearest-rounding
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
_MIDS = (_GRID[1:] + _GRID[:-1]) / 2  # 7 midpoints


def fp8_dequant(w8: torch.Tensor, sinv: torch.Tensor) -> torch.Tensor:
    r, c = w8.shape
    w = w8.to(torch.float32)
    br, bc = sinv.shape
    s = sinv.repeat_interleave(128, 0)[:r].repeat_interleave(128, 1)[:, :c]
    return w * s


def mxfp4_quant(w: torch.Tensor):
    """w [r, c] f32 -> (packed uint8 [r, c/2], e8m0 uint8 [r, c/32])."""
    r, c = w.shape
    g = w.reshape(r, c // 32, 32)
    amax = g.abs().amax(dim=2)
    e = torch.floor(torch.log2(amax.clamp(min=1e-38))) - 2.0
    e = torch.where(amax > 0, e, torch.full_like(e, -127.0)).clamp(-127.0, 127.0)
    scale = torch.exp2(e)
    q = (g / scale.unsqueeze(2)).clamp(-6.0, 6.0)
    mag = q.abs()
    idx = torch.bucketize(mag, _MIDS)  # 0..7 on the e2m1 grid
    nib = idx.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
    nib = torch.where((idx == 0).to(torch.bool), torch.zeros_like(nib), nib)  # -0 -> +0
    nib = nib.reshape(r, c)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    e8m0 = (e + 127.0).round().clamp(0, 254).to(torch.uint8).contiguous()
    return packed, e8m0


def mxfp4_dequant(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    r = packed.shape[0]
    lo = packed & 0xF
    hi = packed >> 4
    nib = torch.stack([lo, hi], dim=2).reshape(r, -1).long()
    mag = _GRID[nib & 7]
    val = torch.where((nib >> 3) == 1, -mag, mag)
    scale = torch.exp2(e8m0.to(torch.float32) - 127.0).repeat_interleave(32, 1)
    return val * scale


idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
wmap = idx["weight_map"]
shards = sorted(set(wmap.values()))
new_map = {}
t0 = time.time()
cos_checks = []

max_shards = int(os.environ.get("MAX_SHARDS", "0")) or len(shards)
for si, shard in enumerate(shards[:max_shards]):
    out_name = shard
    out_tensors = {}
    with safe_open(f"{SRC}/{shard}", framework="pt", device="cpu") as f:
        names = list(f.keys())
        # group expert weights with their scale_inv (same shard in GLM layout)
        pending_scales = {}
        for name in names:
            m = EXP_RE.match(name)
            if not m:
                out_tensors[name] = f.get_tensor(name)
                new_map[name] = out_name
                continue
            kind = m.group(4)
            if kind == "weight_scale_inv":
                continue  # consumed with its weight below
            sname = name + "_scale_inv"
            sinv_name = name.replace(".weight", ".weight_scale_inv")
            w8 = f.get_tensor(name).to(DEV)
            try:
                sinv = f.get_tensor(sinv_name).to(DEV)
            except Exception:
                # scale in another shard
                other = wmap[sinv_name]
                with safe_open(f"{SRC}/{other}", framework="pt", device="cpu") as f2:
                    sinv = f2.get_tensor(sinv_name).to(DEV)
            wf = fp8_dequant(w8, sinv.to(torch.float32))
            packed, e8m0 = mxfp4_quant(wf)
            out_tensors[name] = packed.cpu()
            scale_name = name.replace(".weight", ".weight_scale")
            out_tensors[scale_name] = e8m0.cpu()
            new_map[name] = out_name
            new_map[scale_name] = out_name
            if si % 5 == 0 and len(cos_checks) < 40 and ".experts.0." in name:
                deq = mxfp4_dequant(packed, e8m0)
                cs = torch.nn.functional.cosine_similarity(
                    deq.flatten(), wf.flatten(), dim=0
                ).item()
                cos_checks.append(cs)
            del w8, sinv, wf, packed, e8m0
    save_file(out_tensors, f"{DST}/{out_name}")
    del out_tensors
    torch.cuda.empty_cache()
    el = time.time() - t0
    print(
        f"[{si+1}/{len(shards)}] {shard} done ({el/60:.1f} min elapsed)",
        flush=True,
    )

json.dump(
    {"metadata": idx.get("metadata", {}), "weight_map": new_map},
    open(f"{DST}/model.safetensors.index.json", "w"),
)
print(
    f"DONE in {(time.time()-t0)/60:.1f} min; requant cos min/mean: "
    f"{min(cos_checks):.5f}/{sum(cos_checks)/len(cos_checks):.5f}",
    flush=True,
)
