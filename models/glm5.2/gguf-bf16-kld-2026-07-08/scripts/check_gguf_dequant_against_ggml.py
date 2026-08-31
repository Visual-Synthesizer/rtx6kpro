#!/usr/bin/env python3
"""Compare Python gguf.dequantize() with llama.cpp/GGML dequantization.

The script samples real GGUF tensor blocks, compiles a tiny C helper that calls
GGML's dequantize_row_* functions, and checks for bit-identical float32 output.
It is intentionally focused on the GGML types present in the GLM-5.2 Unsloth
UD-Q4_K_XL checkpoint: Q8_0, Q4_K, Q5_K, and Q6_K.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import gguf
import numpy as np
from gguf.constants import GGML_QUANT_SIZES


C_SOURCE = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-quants.h"

static long file_size(FILE * f) {
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long n = ftell(f);
    if (n < 0) return -1;
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    return n;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s q4_K|q5_K|q6_K|q8_0 input.bin output.f32\n", argv[0]);
        return 2;
    }

    const char * typ = argv[1];
    FILE * in = fopen(argv[2], "rb");
    if (!in) {
        perror("open input");
        return 1;
    }
    long nbytes_l = file_size(in);
    if (nbytes_l <= 0) {
        fprintf(stderr, "bad input size\n");
        fclose(in);
        return 1;
    }
    size_t nbytes = (size_t) nbytes_l;
    uint8_t * data = malloc(nbytes);
    if (!data) {
        fprintf(stderr, "malloc input failed\n");
        fclose(in);
        return 1;
    }
    if (fread(data, 1, nbytes, in) != nbytes) {
        perror("read input");
        free(data);
        fclose(in);
        return 1;
    }
    fclose(in);

    size_t type_size = 0;
    int block = 0;
    if (strcmp(typ, "q4_K") == 0) {
        type_size = sizeof(block_q4_K);
        block = QK_K;
    } else if (strcmp(typ, "q5_K") == 0) {
        type_size = sizeof(block_q5_K);
        block = QK_K;
    } else if (strcmp(typ, "q6_K") == 0) {
        type_size = sizeof(block_q6_K);
        block = QK_K;
    } else if (strcmp(typ, "q8_0") == 0) {
        type_size = sizeof(block_q8_0);
        block = QK8_0;
    } else {
        fprintf(stderr, "unknown type: %s\n", typ);
        free(data);
        return 2;
    }

    if (nbytes % type_size != 0) {
        fprintf(stderr, "input size %zu not multiple of type size %zu\n", nbytes, type_size);
        free(data);
        return 1;
    }

    const size_t nblocks = nbytes / type_size;
    const int64_t k = (int64_t) nblocks * block;
    float * out = malloc((size_t) k * sizeof(float));
    if (!out) {
        fprintf(stderr, "malloc output failed\n");
        free(data);
        return 1;
    }

    if (strcmp(typ, "q4_K") == 0) {
        dequantize_row_q4_K((const block_q4_K *) data, out, k);
    } else if (strcmp(typ, "q5_K") == 0) {
        dequantize_row_q5_K((const block_q5_K *) data, out, k);
    } else if (strcmp(typ, "q6_K") == 0) {
        dequantize_row_q6_K((const block_q6_K *) data, out, k);
    } else {
        dequantize_row_q8_0((const block_q8_0 *) data, out, k);
    }

    FILE * fo = fopen(argv[3], "wb");
    if (!fo) {
        perror("open output");
        free(out);
        free(data);
        return 1;
    }
    if (fwrite(out, sizeof(float), (size_t) k, fo) != (size_t) k) {
        perror("write output");
        fclose(fo);
        free(out);
        free(data);
        return 1;
    }
    fclose(fo);
    free(out);
    free(data);
    return 0;
}
'''


DEFAULT_TENSORS = OrderedDict(
    [
        ("Q8_0", "blk.0.attn_k_b.weight"),
        ("Q4_K", "blk.10.ffn_gate_exps.weight"),
        ("Q5_K", "blk.10.ffn_down_exps.weight"),
        ("Q6_K", "blk.8.ffn_down_exps.weight"),
    ]
)
ARGMAP = {"Q8_0": "q8_0", "Q4_K": "q4_K", "Q5_K": "q5_K", "Q6_K": "q6_K"}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_llama_cpp(path: Path, repo: str, ref: str) -> str:
    if not (path / ".git").exists():
        run(["git", "clone", "--depth", "1", repo, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=path)
    run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=path)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def compile_helper(llama_cpp: Path, build_dir: Path) -> Path:
    c_path = build_dir / "ggml_dequant_ref.c"
    exe = build_dir / "ggml_dequant_ref"
    c_path.write_text(C_SOURCE)
    cmd = [
        "gcc",
        "-O2",
        "-std=c11",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wl,--gc-sections",
        "-D_GNU_SOURCE",
        "-D_DEFAULT_SOURCE",
        "-DGGML_VERSION=0",
        '-DGGML_COMMIT="local-check"',
        "-I" + str(llama_cpp / "ggml/include"),
        "-I" + str(llama_cpp / "ggml/src"),
        str(c_path),
        str(llama_cpp / "ggml/src/ggml-quants.c"),
        "-lm",
        "-o",
        str(exe),
    ]
    run(cmd)
    return exe


def find_tensor(gguf_dir: Path, name: str):
    for path in sorted(gguf_dir.glob("*.gguf")):
        reader = gguf.GGUFReader(str(path))
        for tensor in reader.tensors:
            if tensor.name == name:
                return path, tensor
    raise KeyError(name)


def compare_one(exe: Path, tensor, type_name: str, offset: int, blocks: int) -> dict:
    qtype = tensor.tensor_type
    _, type_size = GGML_QUANT_SIZES[qtype]
    raw = np.ascontiguousarray(tensor.data.view(np.uint8).reshape(-1))
    sample = np.ascontiguousarray(raw[offset * type_size : (offset + blocks) * type_size])
    py = gguf.dequantize(sample.reshape(blocks, type_size), qtype).astype(
        np.float32, copy=False
    ).reshape(-1)

    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "sample.bin"
        out = Path(td) / "sample.f32"
        sample.tofile(inp)
        run([str(exe), ARGMAP[type_name], str(inp), str(out)])
        c = np.fromfile(out, dtype=np.float32)

    diff = np.abs(py - c)
    return {
        "offset_block": int(offset),
        "sample_blocks": int(blocks),
        "sample_values": int(py.size),
        "bit_equal": bool(np.array_equal(py.view(np.uint32), c.view(np.uint32))),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "nonzero_diffs": int(np.count_nonzero(diff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-dir", required=True)
    parser.add_argument("--llama-cpp", default="/tmp/llama.cpp-ggml-check")
    parser.add_argument(
        "--llama-ref",
        default="a646006f09d2f76f2d62d6c0d5e8e8490d570720",
        help="llama.cpp commit/ref used for the GGML dequantization reference",
    )
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    gguf_dir = Path(args.gguf_dir)
    llama_cpp = Path(args.llama_cpp)
    commit = ensure_llama_cpp(
        llama_cpp, "https://github.com/ggml-org/llama.cpp", args.llama_ref
    )

    with tempfile.TemporaryDirectory() as td:
        exe = compile_helper(llama_cpp, Path(td))
        results: list[dict] = []
        for type_name, tensor_name in DEFAULT_TENSORS.items():
            gguf_file, tensor = find_tensor(gguf_dir, tensor_name)
            if tensor.tensor_type.name != type_name:
                raise RuntimeError(
                    f"{tensor_name}: expected {type_name}, found {tensor.tensor_type.name}"
                )
            _, type_size = GGML_QUANT_SIZES[tensor.tensor_type]
            total_blocks = tensor.data.view(np.uint8).size // type_size
            blocks = min(args.blocks, total_blocks)
            offsets = sorted(
                set(
                    [
                        0,
                        max(0, total_blocks // 2 - blocks // 2),
                        max(0, total_blocks - blocks),
                    ]
                )
            )
            for offset in offsets:
                row = {
                    "type": type_name,
                    "tensor": tensor_name,
                    "file": gguf_file.name,
                    "llama_cpp_commit": commit,
                }
                row.update(compare_one(exe, tensor, type_name, offset, blocks))
                results.append(row)

    payload = {"llama_cpp_commit": commit, "results": results}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")

    if not all(row["bit_equal"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
