#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import shutil
from pathlib import Path

import gguf
import torch
from safetensors.torch import save_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def nbytes_bf16(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n * 2


def to_bf16_tensor(tensor):
    weight_type = tensor.tensor_type
    data = tensor.data
    if weight_type.name == "BF16" and data.dtype.name == "uint8":
        x = data.view("uint16")
        return torch.from_numpy(x).view(torch.bfloat16).clone()
    if weight_type.name in ("F32", "F16", "BF16"):
        return torch.from_numpy(data).to(torch.bfloat16).contiguous()
    x = gguf.dequantize(data, weight_type)
    return torch.from_numpy(x).to(torch.bfloat16).contiguous()


def layer_from_name(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def simple_map(name):
    if name == "output.weight":
        return "lm_head.weight"
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"

    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    i = int(m.group(1))
    rest = m.group(2)
    p = f"model.layers.{i}"
    table = {
        "attn_norm.weight": f"{p}.input_layernorm.weight",
        "ffn_norm.weight": f"{p}.post_attention_layernorm.weight",
        "attn_q_a.weight": f"{p}.self_attn.q_a_proj.weight",
        "attn_q_b.weight": f"{p}.self_attn.q_b_proj.weight",
        "attn_kv_a_mqa.weight": f"{p}.self_attn.kv_a_proj_with_mqa.weight",
        "attn_kv_a_norm.weight": f"{p}.self_attn.kv_a_layernorm.weight",
        "attn_q_a_norm.weight": f"{p}.self_attn.q_a_layernorm.weight",
        "attn_output.weight": f"{p}.self_attn.o_proj.weight",
        "indexer.attn_k.weight": f"{p}.self_attn.indexer.wk.weight",
        "indexer.attn_q_b.weight": f"{p}.self_attn.indexer.wq_b.weight",
        "indexer.proj.weight": f"{p}.self_attn.indexer.weights_proj.weight",
        "indexer.k_norm.weight": f"{p}.self_attn.indexer.k_norm.weight",
        "indexer.k_norm.bias": f"{p}.self_attn.indexer.k_norm.bias",
        "ffn_gate.weight": f"{p}.mlp.gate_proj.weight",
        "ffn_up.weight": f"{p}.mlp.up_proj.weight",
        "ffn_down.weight": f"{p}.mlp.down_proj.weight",
        "ffn_gate_inp.weight": f"{p}.mlp.gate.weight",
        "exp_probs_b.bias": f"{p}.mlp.gate.e_score_correction_bias",
        "ffn_gate_shexp.weight": f"{p}.mlp.shared_experts.gate_proj.weight",
        "ffn_up_shexp.weight": f"{p}.mlp.shared_experts.up_proj.weight",
        "ffn_down_shexp.weight": f"{p}.mlp.shared_experts.down_proj.weight",
        "nextn.eh_proj.weight": f"{p}.eh_proj.weight",
        "nextn.enorm.weight": f"{p}.enorm.weight",
        "nextn.hnorm.weight": f"{p}.hnorm.weight",
        "nextn.shared_head_norm.weight": f"{p}.shared_head.norm.weight",
    }
    return table.get(rest)


def expert_prefix(name):
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
    if not m:
        return None
    layer = int(m.group(1))
    kind = m.group(2)
    proj = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[kind]
    return layer, proj


def shard_name(seq):
    return f"model-{seq:06d}.safetensors"


def write_shard(out_dir, seq, tensors, weight_map, total_size):
    fn = shard_name(seq)
    path = out_dir / fn
    tmp = out_dir / f".{fn}.tmp"
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
    for name, tensor in tensors.items():
        weight_map[name] = fn
        total_size[0] += tensor.numel() * tensor.element_size()
    return fn


def copy_sidecars(src, dst):
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def main():
    args = parse_args()
    gguf_dir = Path(args.gguf_dir)
    out_dir = Path(args.out_dir)
    config_dir = Path(args.config_dir or args.gguf_dir)

    files = sorted(gguf_dir.glob("*.gguf"))
    if not files:
        raise SystemExit(f"No GGUF files found in {gguf_dir}")

    readers = [gguf.GGUFReader(str(p)) for p in files]
    by_name = {}
    for reader in readers:
        for tensor in reader.tensors:
            by_name[tensor.name] = tensor

    if args.dry_run:
        total = 0
        skipped = []
        for name, tensor in by_name.items():
            if name.endswith(".attn_k_b.weight") or name.endswith(".attn_v_b.weight"):
                continue
            if expert_prefix(name):
                layer, proj = expert_prefix(name)
                experts, rows, cols = tensor.shape[2], tensor.shape[1], tensor.shape[0]
                if proj == "down_proj":
                    total += nbytes_bf16((experts, cols, rows))
                else:
                    total += nbytes_bf16((experts, rows, cols))
                continue
            mapped = simple_map(name)
            if mapped:
                total += nbytes_bf16(reversed(tuple(tensor.shape)))
            else:
                skipped.append(name)
        layers = sorted({layer_from_name(n) for n in by_name if layer_from_name(n) is not None})
        for layer in layers:
            k = by_name.get(f"blk.{layer}.attn_k_b.weight")
            v = by_name.get(f"blk.{layer}.attn_v_b.weight")
            if k is not None and v is not None:
                heads = int(k.shape[2])
                kv = int(k.shape[1])
                qk = int(k.shape[0])
                vd = int(v.shape[1])
                total += nbytes_bf16((heads * (qk + vd), kv))
        print(json.dumps({
            "gguf_files": len(files),
            "gguf_tensors": len(by_name),
            "estimated_bf16_bytes": total,
            "estimated_bf16_tib": total / 1024**4,
            "skipped": skipped,
        }, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(config_dir, out_dir)

    weight_map = {}
    total_size = [0]
    seq = 1

    done = set()

    def log(msg):
        print(msg, flush=True)

    # Normal non-expert tensors.
    for name in sorted(by_name):
        if (
            name.endswith(".attn_k_b.weight")
            or name.endswith(".attn_v_b.weight")
            or expert_prefix(name)
        ):
            continue
        mapped = simple_map(name)
        if mapped is None:
            log(f"SKIP {name}")
            continue
        if mapped in done:
            continue
        log(f"[{seq:06d}] {name} -> {mapped}")
        tensor = to_bf16_tensor(by_name[name])
        write_shard(out_dir, seq, {mapped: tensor}, weight_map, total_size)
        done.add(mapped)
        seq += 1
        del tensor
        gc.collect()

    # Combine split GGUF K/V B projection into HF kv_b_proj.weight.
    layers = sorted({layer_from_name(n) for n in by_name if layer_from_name(n) is not None})
    for layer in layers:
        k_name = f"blk.{layer}.attn_k_b.weight"
        v_name = f"blk.{layer}.attn_v_b.weight"
        if k_name not in by_name and v_name not in by_name:
            continue
        if k_name not in by_name or v_name not in by_name:
            raise RuntimeError(f"Missing split kv_b pair for layer {layer}")
        mapped = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        log(f"[{seq:06d}] {k_name}+{v_name} -> {mapped}")
        k = to_bf16_tensor(by_name[k_name]).permute(0, 2, 1).contiguous()
        v = to_bf16_tensor(by_name[v_name]).contiguous()
        kv = torch.cat([k, v], dim=1).reshape(-1, k.shape[-1]).contiguous()
        write_shard(out_dir, seq, {mapped: kv}, weight_map, total_size)
        done.add(mapped)
        seq += 1
        del k, v, kv
        gc.collect()

    # Split merged GGUF MoE tensors into standard HF expert keys.
    for name in sorted(by_name):
        info = expert_prefix(name)
        if info is None:
            continue
        layer, proj = info
        log(f"[{seq:06d}] {name} -> model.layers.{layer}.mlp.experts.*.{proj}.weight")
        dense = to_bf16_tensor(by_name[name])
        tensors = {
            f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight": dense[expert]
            for expert in range(dense.shape[0])
        }
        write_shard(out_dir, seq, tensors, weight_map, total_size)
        done.update(tensors)
        seq += 1
        del dense, tensors
        gc.collect()

    index = {
        "metadata": {"total_size": total_size[0]},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    log(f"DONE shards={seq - 1} tensors={len(weight_map)} total_size={total_size[0]}")


if __name__ == "__main__":
    main()
