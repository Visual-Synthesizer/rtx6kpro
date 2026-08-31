# GLM-5.1 v8 DCP Graph Fix on 8x RTX PRO 6000 Blackwell

Measured on 2026-06-06 on the local 16-GPU RTX PRO 6000 Blackwell host.

Status: DCP sparse MLA correctness fix is validated and PR is open. Full
30-second decode matrices are recorded for A16-on and A16-off, DCP1/2/4/8,
MTP off/on.

## Image And Source

Runtime image used for the measurements:

```text
voipmonitor/vllm:cu132-vllm611a842-b12xf9226c-a16nativew4a16-20260606
```

The benchmark used this image with a single source overlay for the DCP graph fix:

```text
/root/vllm/worktrees/vllm-main-dcp-sparse-cg-fix-20260606/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
```

Patch PR:

```text
https://github.com/local-inference-lab/vllm/pull/3
```

Relevant source state:

| Component | Revision |
|---|---|
| CUDA | `13.2.1` |
| cuBLAS | `13.4.1.2-1` |
| cuDNN | `9.22.0.52-1` |
| NCCL | `2.30.4` |
| PyTorch | `2.12.0+cu132` |
| vLLM image branch | `dev/abyssal-abjuration` |
| vLLM image commit | `611a842dc1052772b22e5ac48b2da28ced6dfba9` |
| vLLM overlay branch | `codex/glm51-dcp-sparse-cg-fix-20260606` |
| vLLM overlay commit | `6eeabd7616357f93e2fd83637e7d789b5fafc62e` |
| B12X repo | `https://github.com/lukealonso/b12x.git` |
| B12X commit | `f9226c99384b8f7a169e6f5d6251f783886ef775` |
| FlashInfer commit | `e8d31317bedb4efd52559a2234f4cb9e83428cb9` |

## Runtime

Model:

```text
/mnt/glm51-luke-nvfp4-mtp-nvfp4routed-symlink
```

Served name:

```text
GLM-5.1
```

Launcher used for the sweep:

```text
/root/bench-results/glm51-v8-patchedcg-20260606/run_variant.sh
```

Important runtime settings:

| Setting | Value |
|---|---|
| TP | `8` |
| DCP | `1`, `2`, `4`, `8` |
| MTP | off/on, `num_speculative_tokens=3`, probabilistic |
| Quantization | `modelopt_fp4` |
| Attention | `B12X_MLA_SPARSE` |
| MoE | `b12x` |
| KV cache | `fp8` |
| A16-on flag | `B12X_MOE_FORCE_A16=1` |
| TC decode flag | `B12X_W4A16_TC_DECODE=0` |
| Max model len | not explicitly limited; runtime reports `202,752` |
| Max batched tokens | `16,384` |
| Max seqs | `64` |
| CUDA graph capture | no-MTP `64`, MTP `256` |

The launcher deliberately unsets these before serving:

```bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_B12X_FORCE_MOE_A16
```

Example single-variant run:

```bash
DCP_SIZE=4 MTP=1 A16=1 PORT=5341 CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  /root/bench-results/glm51-v8-patchedcg-20260606/run_variant.sh
```

## Correctness Fix

Root cause: B12X sparse MLA reported CUDA graph support for DCP>1, so vLLM
captured FULL_AND_PIECEWISE graphs. DCP sparse MLA is not full-graph-stable yet,
and full graph replay corrupted GLM-5.1 long-context generation.

Fix: for `decode_context_parallel_size > 1`, B12X sparse MLA now reports
`AttentionCGSupport.NEVER`. vLLM then downgrades to PIECEWISE under
VLLM_COMPILE. DCP1 remains on the normal graph-support path.

Validation:

| Profile | Result |
|---|---|
| DCP2 no-MTP A16-on | short smoke coherent/CJK 0; `-L -c10000` coherent for 3 iterations |
| DCP4 no-MTP A16-on | short smoke coherent/CJK 0; `-L -c10000` coherent for multiple iterations |
| DCP4 MTP A16-on | short smoke coherent/CJK 0; `-L -c10000` coherent for 6 iterations |
| DCP8 no-MTP A16-on | short smoke coherent/CJK 0; `-L -c10000` coherent for 5 iterations |

## Benchmark Command

Each matrix used:

```bash
python3 /root/llm-inference-bench/llm_decode_bench.py \
  --port PORT \
  --model GLM-5.1 \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 0,16k,32k,64k,128k \
  --duration 30 \
  --max-tokens 8192 \
  --display-mode plain \
  --no-hw-monitor \
  --output result.json
```

Hardware monitoring was disabled because two 8-GPU servers were benchmarked in
parallel on the 16-GPU host.

Result directory:

```text
/root/bench-results/glm51-v8-patchedcg-20260606
```

## A16-On Decode Matrices

### DCP1, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 69.0 | 115.7 | 201.2 | 306.4 | 456.7 | 636.3 | ∅ |
| 16k | 66.7 | 111.2 | 198.2 | 312.5 | 484.9 | ∅ | ∅ |
| 32k | 65.1 | 111.0 | 193.6 | 301.9 | ∅ | ∅ | ∅ |
| 64k | 65.0 | 108.7 | 188.1 | ∅ | ∅ | ∅ | ∅ |
| 128k | 63.2 | 104.7 | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP1, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 85.4 | 149.3 | 259.9 | 429.8 | 651.9 | 984.2 | ∅ |
| 16k | 78.7 | 139.0 | 233.4 | 376.7 | ∅ | ∅ | ∅ |
| 32k | 77.0 | 132.8 | 221.6 | 361.4 | ∅ | ∅ | ∅ |
| 64k | 75.4 | 126.1 | 214.4 | ∅ | ∅ | ∅ | ∅ |
| 128k | 70.6 | 117.3 | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP2, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.5 | 19.7 | 36.4 | 39.2 | 78.1 | 158.7 | 310.3 |
| 16k | 10.6 | 19.2 | 39.1 | 38.9 | 77.3 | ∅ | ∅ |
| 32k | 10.7 | 19.3 | 38.5 | 38.6 | 77.4 | ∅ | ∅ |
| 64k | 10.5 | 19.4 | 38.7 | 38.4 | ∅ | ∅ | ∅ |
| 128k | 10.5 | 19.4 | 38.6 | ∅ | ∅ | ∅ | ∅ |

### DCP2, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.5 | 24.5 | 48.9 | 95.1 | 184.2 | 390.4 | 768.3 |
| 16k | 28.1 | 24.6 | 48.9 | 98.7 | 202.4 | ∅ | ∅ |
| 32k | 28.2 | 24.4 | 47.3 | 102.0 | 195.5 | ∅ | ∅ |
| 64k | 26.9 | 24.0 | 51.0 | 101.9 | ∅ | ∅ | ∅ |
| 128k | 27.0 | 26.0 | 53.3 | ∅ | ∅ | ∅ | ∅ |

### DCP4, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.7 | 20.7 | 40.7 | 55.9 | 108.9 | 218.3 | 418.3 |
| 16k | 10.7 | 19.5 | 39.4 | 54.8 | 108.6 | 217.0 | 426.7 |
| 32k | 10.6 | 19.6 | 39.7 | 54.3 | 108.0 | 213.0 | ∅ |
| 64k | 10.8 | 19.4 | 39.6 | 53.8 | 107.7 | ∅ | ∅ |
| 128k | 10.7 | 19.6 | 39.1 | 53.9 | ∅ | ∅ | ∅ |

### DCP4, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27.1 | 39.3 | 70.3 | 141.6 | 267.9 | 542.4 | 1028.7 |
| 16k | 27.0 | 35.1 | 69.2 | 138.5 | 270.7 | 532.6 | ∅ |
| 32k | 27.6 | 33.9 | 73.1 | 134.5 | 272.1 | 528.6 | ∅ |
| 64k | 27.1 | 36.3 | 69.6 | 140.3 | 268.9 | ∅ | ∅ |
| 128k | 27.7 | 36.0 | 71.1 | 135.7 | ∅ | ∅ | ∅ |

### DCP8, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.6 | 19.5 | 38.5 | 68.3 | 134.8 | 261.5 | 341.2 |
| 16k | 10.6 | 19.7 | 39.6 | 67.2 | 134.7 | 267.6 | 454.4 |
| 32k | 10.4 | 19.7 | 40.7 | 67.1 | 133.0 | 269.8 | 528.3 |
| 64k | 10.5 | 19.9 | 40.4 | 67.3 | 135.5 | 265.9 | ∅ |
| 128k | 10.4 | 20.2 | 39.5 | 67.4 | 134.0 | ∅ | ∅ |

### DCP8, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27.8 | 44.6 | 86.9 | 169.2 | 348.6 | 648.4 | 962.0 |
| 16k | 27.5 | 43.1 | 89.0 | 171.5 | 339.1 | 549.8 | 796.2 |
| 32k | 25.4 | 44.7 | 92.4 | 178.7 | 349.9 | 501.6 | ∅ |
| 64k | 28.4 | 44.0 | 91.7 | 173.1 | 339.4 | ∅ | ∅ |
| 128k | 27.0 | 44.0 | 87.3 | 182.3 | ∅ | ∅ | ∅ |

## A16-On Prefill Summary

| Variant | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| DCP1 MTP off | 3,622 | 3,352 | 3,019 | 2,483 | 1,842 |
| DCP1 MTP on | 3,532 | 3,277 | 2,907 | 2,389 | 1,757 |
| DCP2 MTP off | 3,612 | 3,405 | 3,225 | 2,876 | 2,400 |
| DCP2 MTP on | 3,456 | 3,272 | 3,093 | 2,754 | 2,291 |
| DCP4 MTP off | 3,071 | 2,691 | 2,663 | 2,488 | 2,270 |
| DCP4 MTP on | 2,955 | 2,720 | 2,564 | 2,391 | 2,179 |
| DCP8 MTP off | 2,230 | 2,091 | 1,886 | 1,743 | 1,646 |
| DCP8 MTP on | 2,163 | 2,028 | 1,826 | 1,686 | 1,590 |

## A16-Off Decode Matrices

### DCP1, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75.2 | 121.6 | 211.7 | 311.9 | 476.9 | 652.4 | 862.0 |
| 16k | 72.6 | 117.9 | 207.8 | 317.6 | 503.0 | ∅ | ∅ |
| 32k | 70.8 | 116.6 | 204.8 | 305.1 | ∅ | ∅ | ∅ |
| 64k | 70.4 | 114.1 | 195.7 | ∅ | ∅ | ∅ | ∅ |
| 128k | 68.3 | 109.6 | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP1, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 85.7 | 147.9 | 265.4 | 441.9 | 707.5 | 820.4 | ∅ |
| 16k | 81.8 | 143.7 | 240.2 | 389.0 | 599.2 | ∅ | ∅ |
| 32k | 82.5 | 133.5 | 235.1 | 366.5 | ∅ | ∅ | ∅ |
| 64k | 78.6 | 126.9 | 217.9 | ∅ | ∅ | ∅ | ∅ |
| 128k | 77.4 | 117.2 | ∅ | ∅ | ∅ | ∅ | ∅ |

### DCP2, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.6 | 19.3 | 35.6 | 40.0 | 79.6 | 159.4 | 317.9 |
| 16k | 10.7 | 19.5 | 39.0 | 39.3 | 78.7 | 157.1 | ∅ |
| 32k | 10.6 | 19.6 | 39.4 | 39.7 | 79.1 | ∅ | ∅ |
| 64k | 10.7 | 19.7 | 39.5 | 39.5 | ∅ | ∅ | ∅ |
| 128k | 10.7 | 19.3 | 39.1 | ∅ | ∅ | ∅ | ∅ |

### DCP2, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.8 | 24.2 | 47.2 | 97.3 | 191.3 | 376.3 | 767.8 |
| 16k | 27.0 | 25.1 | 50.1 | 97.1 | 189.7 | 404.5 | ∅ |
| 32k | 28.2 | 25.7 | 49.8 | 97.3 | 205.9 | ∅ | ∅ |
| 64k | 27.5 | 25.9 | 49.3 | 98.7 | ∅ | ∅ | ∅ |
| 128k | 28.7 | 25.2 | 53.8 | ∅ | ∅ | ∅ | ∅ |

### DCP4, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.6 | 19.8 | 36.9 | 54.4 | 108.8 | 218.7 | 425.8 |
| 16k | 10.7 | 19.8 | 39.5 | 54.7 | 108.2 | 216.6 | 434.6 |
| 32k | 10.7 | 19.9 | 39.8 | 54.0 | 108.7 | 215.7 | ∅ |
| 64k | 10.7 | 20.0 | 39.8 | 54.0 | 109.1 | ∅ | ∅ |
| 128k | 10.7 | 20.0 | 39.8 | 54.0 | ∅ | ∅ | ∅ |

### DCP4, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.4 | 36.4 | 67.8 | 131.5 | 273.7 | 525.2 | 987.6 |
| 16k | 26.2 | 35.3 | 71.0 | 143.8 | 272.3 | 526.4 | 786.6 |
| 32k | 26.7 | 35.5 | 67.8 | 138.4 | 249.6 | 503.8 | ∅ |
| 64k | 28.6 | 31.5 | 63.6 | 129.2 | 258.5 | ∅ | ∅ |
| 128k | 27.0 | 31.7 | 62.8 | 136.4 | ∅ | ∅ | ∅ |

### DCP8, MTP Off

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.8 | 19.9 | 35.1 | 67.0 | 134.6 | 266.9 | 346.0 |
| 16k | 10.7 | 20.1 | 39.7 | 67.7 | 136.1 | 267.3 | 463.9 |
| 32k | 10.7 | 20.6 | 40.9 | 67.5 | 135.2 | 267.2 | 536.3 |
| 64k | 10.8 | 20.3 | 41.3 | 69.2 | 135.1 | 274.8 | ∅ |
| 128k | 10.8 | 20.0 | 40.6 | 68.6 | 136.6 | ∅ | ∅ |

### DCP8, MTP On

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25.2 | 44.8 | 82.7 | 163.8 | 338.1 | 565.2 | 847.1 |
| 16k | 27.2 | 43.5 | 87.5 | 173.0 | 332.4 | 484.9 | 701.9 |
| 32k | 27.5 | 43.7 | 85.9 | 165.5 | 326.7 | 448.1 | 643.7 |
| 64k | 27.4 | 41.0 | 83.9 | 171.6 | 343.4 | 436.0 | ∅ |
| 128k | 26.2 | 45.9 | 93.3 | 185.3 | 358.3 | ∅ | ∅ |

## A16-Off Prefill Summary

| Variant | 8k | 16k | 32k | 64k | 128k |
|---|---:|---:|---:|---:|---:|
| DCP1 MTP off | 4,073 | 3,748 | 3,281 | 2,666 | 1,933 |
| DCP1 MTP on | 3,988 | 3,666 | 3,158 | 2,560 | 1,842 |
| DCP2 MTP off | 4,044 | 3,808 | 3,522 | 3,114 | 2,548 |
| DCP2 MTP on | 3,887 | 3,656 | 3,371 | 2,976 | 2,429 |
| DCP4 MTP off | 3,366 | 3,068 | 2,851 | 2,656 | 2,399 |
| DCP4 MTP on | 3,249 | 2,982 | 2,753 | 2,558 | 2,304 |
| DCP8 MTP off | 2,385 | 2,233 | 1,983 | 1,827 | 1,714 |
| DCP8 MTP on | 2,320 | 2,171 | 1,923 | 1,767 | 1,656 |

## Pending

- Build a final image after PR #3 is merged so the wiki no longer needs the
  source overlay mount.
