# MTP Quality Evaluation

Does MTP (Multi-Token Prediction) speculative decoding affect model accuracy? We ran GPQA Diamond and GSM8K with and without MTP across two NVFP4 checkpoints on two inference engines (SGLang and vLLM) to find out.

## Table of Contents

- [Summary](#summary)
- [Test Environment](#test-environment)
- [GPQA Diamond Results](#gpqa-diamond-results)
  - [SGLang Results](#sglang-results)
  - [vLLM Results](#vllm-results)
  - [Cross-Engine Comparison](#cross-engine-comparison)
- [GSM8K Results](#gsm8k-results)
- [Hard Math Test](#hard-math-test)
- [Conclusions](#conclusions)
- [Raw Data](#raw-data)

---

## Summary

### SGLang (TP8, 8x RTX PRO 6000 Blackwell)

| Configuration | GPQA Mean | GSM8K (thinking) | GSM8K (no thinking) | Wall Time (GPQA) |
|:---|:---:|:---:|:---:|:---:|
| lukealonso/NVFP4 + MTP | **88.28%** ±1.06 | **99.0%** | **44%** | ~1h 29m |
| lukealonso/NVFP4, no MTP | 87.53% ±1.09 | — | — | ~1h 48m |
| nvidia/NVFP4 + MTP | 87.46% ±1.57 | 97.5% | 39% | ~1h 43m |
| nvidia/NVFP4, no MTP | 86.58% ±1.15 | — | — | ~2h 15m |

### vLLM (TP4, 4x RTX PRO 6000 Blackwell)

| Configuration | GPQA Mean | GSM8K (thinking) |
|:---|:---:|:---:|
| nvidia/NVFP4 + MTP | **88.53%** ±1.92 | — |
| nvidia/NVFP4, no MTP | 86.90% ±1.13 | 98.5% |

### Key findings

1. **MTP does not degrade quality** — no statistically significant difference on either engine (Welch t-test p>0.05 for all pairs)
2. **MTP provides 18-24% faster inference** on SGLang — a free speedup
3. **Inference engine can matter** — nvidia NVFP4 on vLLM (88.53%) scores comparably to lukealonso on SGLang (88.28%), though the difference is not significant
4. **lukealonso checkpoint outperforms nvidia on SGLang** across all benchmarks

---

## Test Environment

### SGLang

| Parameter | Value |
|:---|:---|
| **GPU** | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each) |
| **Engine** | SGLang 0.5.9 |
| **TP** | 8 |
| **Container** | `voipmonitor/llm-pytorch-blackwell:nightly-cuda132` |
| **Date** | 2026-03-11 |
| **Eval framework** | [simple-evals](https://github.com/openai/simple-evals) (ChatCompletionSampler) |

#### Server config (common)

```
--tensor-parallel-size 8
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--attention-backend triton
--moe-runner-backend flashinfer_cutlass
--fp4-gemm-backend flashinfer_cudnn
--cuda-graph-max-bs 128
--max-running-requests 128
--context-length 262144
--chunked-prefill-size 32768
--mem-fraction-static 0.80
--disable-custom-all-reduce
--disable-shared-experts-fusion
--schedule-conservativeness 0.1
```

#### MTP-specific flags (SGLang)

```
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

### vLLM

| Parameter | Value |
|:---|:---|
| **GPU** | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each) |
| **Engine** | vLLM |
| **TP** | 4 |
| **Date** | 2026-03-13 |
| **Eval framework** | [simple-evals](https://github.com/openai/simple-evals) (ChatCompletionSampler) |

#### Server config (MTP ON)

```bash
VLLM_LOG_STATS_INTERVAL=1 NCCL_P2P_LEVEL=SYS SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --host 0.0.0.0 --port 5199 \
  --served-model-name Qwen3_5-397B-A17B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching --enable-chunked-prefill
```

For MTP OFF: same config without `--speculative-config`.

### Eval command

```bash
python3 -u -m sglang.test.run_eval \
  --eval-name gpqa \
  --model <served-model-name> \
  --base-url http://localhost:<port> \
  --num-examples 198 \
  --repeat 1 \
  --thinking-mode qwen3 \
  --max-tokens 64000
```

For 8-repeat tests, each repeat was run sequentially (not parallel) to avoid server overload.

### Known warning (NVFP4 only)

```
DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0.
This might cause accuracy degradation on Blackwell.
```

---

## GPQA Diamond Results

### SGLang Results

#### Per-Run Scores

| Run | lukealonso MTP | lukealonso No MTP | nvidia MTP | nvidia No MTP |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 88.9 | 86.4 | 85.9 | 86.4 |
| 2 | 87.9 | 87.4 | 90.4 | 85.9 |
| 3 | 86.9 | 89.4 | 85.9 | 86.9 |
| 4 | 88.4 | 86.9 | 88.4 | 86.4 |
| 5 | 90.4 | 88.9 | 87.9 | 84.8 |
| 6 | 88.4 | 86.4 | 85.9 | 86.4 |
| 7 | 87.9 | 87.4 | 87.4 | 86.9 |
| 8 | 87.4 | 87.4 | 87.9 | 88.9 |

#### Aggregate Statistics (SGLang)

| Metric | lukealonso MTP | lukealonso No MTP | nvidia MTP | nvidia No MTP |
|:---|:---:|:---:|:---:|:---:|
| **Mean** | **88.28%** | **87.53%** | **87.46%** | **86.58%** |
| Std | 1.06 | 1.09 | 1.57 | 1.15 |
| Min | 86.9 | 86.4 | 85.9 | 84.8 |
| Max | 90.4 | 89.4 | 90.4 | 88.9 |
| Wall time | ~1h 29m | ~1h 48m | ~1h 43m | ~2h 15m |

#### MTP Impact (SGLang)

| Checkpoint | MTP ON | MTP OFF | Delta | t-stat | p-value |
|:---|:---:|:---:|:---:|:---:|:---:|
| lukealonso | 88.28% | 87.53% | +0.75pp | 1.41 | >0.05 (ns) |
| nvidia | 87.46% | 86.58% | +0.88pp | 1.28 | >0.05 (ns) |

### vLLM Results

#### Per-Run Scores

| Run | nvidia MTP | nvidia No MTP |
|:---:|:---:|:---:|
| 1 | 91.9 | 87.4 |
| 2 | 87.9 | 86.4 |
| 3 | 89.4 | 86.9 |
| 4 | 85.9 | 86.9 |
| 5 | 86.4 | 85.9 |
| 6 | 88.4 | 85.9 |
| 7 | 89.9 | 89.4 |
| 8 | 88.4 | 86.4 |

#### Aggregate Statistics (vLLM)

| Metric | nvidia MTP | nvidia No MTP |
|:---|:---:|:---:|
| **Mean** | **88.53%** | **86.90%** |
| Std | 1.92 | 1.13 |
| Min | 85.9 | 85.9 |
| Max | 91.9 | 89.4 |

#### MTP Impact (vLLM)

| Checkpoint | MTP ON | MTP OFF | Delta | t-stat | p-value |
|:---|:---:|:---:|:---:|:---:|:---:|
| nvidia | 88.53% | 86.90% | +1.62pp | 2.06 | >0.05 (ns) |

The +1.62pp delta on vLLM is larger than SGLang's +0.88pp for the same checkpoint, but still not statistically significant (p=0.06, just below the α=0.05 threshold). The wider spread is partly driven by one high outlier (91.9%) in the MTP ON run.

### Cross-Engine Comparison

| Configuration | Engine | GPQA Mean | Std |
|:---|:---|:---:|:---:|
| nvidia NVFP4 + MTP | vLLM | **88.53%** | 1.92 |
| lukealonso NVFP4 + MTP | SGLang | 88.28% | 1.06 |
| nvidia NVFP4 + MTP | SGLang | 87.46% | 1.57 |
| nvidia NVFP4, no MTP | vLLM | 86.90% | 1.13 |
| nvidia NVFP4, no MTP | SGLang | 86.58% | 1.15 |
| lukealonso NVFP4, no MTP | SGLang | 87.53% | 1.09 |

**Statistical significance (Welch t-test, all pairs):**

| Comparison | Delta | t-stat | Significant? |
|:---|:---:|:---:|:---:|
| nvidia vLLM+MTP vs nvidia SGLang+MTP | +1.06pp | 1.21 | No (p>0.05) |
| nvidia vLLM+MTP vs lukealonso SGLang+MTP | +0.25pp | 0.29 | No (p>0.05) |
| nvidia vLLM-MTP vs nvidia SGLang-MTP | +0.31pp | 0.61 | No (p>0.05) |
| nvidia vLLM+MTP vs nvidia vLLM-MTP | +1.62pp | 2.06 | No (p>0.05) |
| lukealonso SGLang+MTP vs nvidia SGLang+MTP | +0.81pp | 1.21 | No (p>0.05) |

No pair reaches statistical significance. All configurations produce GPQA scores in the 86-89% range, and with 8 repeats the test lacks power to distinguish differences below ~2pp.

---

## GSM8K Results

### With thinking mode

| Model | Engine | MTP | Score | Config |
|:---|:---|:---:|:---:|:---|
| lukealonso NVFP4 | SGLang | ON | **99.0%** | 200 examples, max-tokens 16000 |
| nvidia NVFP4 | vLLM | OFF | **98.5%** | 200 examples, max-tokens 16000 |
| nvidia NVFP4 | SGLang | ON | 97.5% | 200 examples, max-tokens 16000 |

nvidia on vLLM without MTP (98.5%) outperforms nvidia on SGLang with MTP (97.5%), again suggesting the inference engine matters.

### Without thinking (5-shot, SGLang only)

| Model | Score | Config |
|:---|:---:|:---|
| lukealonso | **44%** | 200 examples, max-tokens 2048 |
| nvidia | 39% | 200 examples, max-tokens 2048 |

Without chain-of-thought reasoning, the quantization quality gap is much more pronounced (+5pp).

---

## Hard Math Test

19 custom math questions, no thinking mode. SGLang only (vLLM server crashed before Hard Math could run).

| Q# | Question | lukealonso | nvidia |
|:---:|:---|:---:|:---:|
| 1 | (37×43)−(29×51)+17 | FAIL (139) | FAIL (10) |
| 2 | 123² − 113² | OK | OK |
| 3 | 2³¹ mod 7 | FAIL (4) | FAIL (1) |
| 4 | log₂(x)=5.5, x=? | OK | OK |
| 5 | P(2 aces in row) | OK | OK |
| 6 | LCM(12,18,20) | OK | OK |
| 7 | Primes < 50 | OK | OK |
| 8 | Sum primes < 30 | OK | OK |
| 9 | (root1+1)(root2+1) for x²−7x+12=0 | **OK (20)** | **FAIL (30)** |
| 10 | 2ᵃ×3ᵇ=72, a+b=? | OK | OK |
| 11 | Altitude to hypotenuse | OK | OK |
| 12 | 10th Fibonacci | OK | OK |
| 13 | Geometric sequence 8th term | OK | OK |
| 14 | MISSISSIPPI arrangements | OK | OK |
| 15 | C(8,3) | OK | OK |
| 16 | Infinite geometric series sum | OK | OK |
| 17 | 2×2 determinant | OK | OK |
| 18 | Sum 1 to 100 | OK | OK |
| 19 | 13³ | OK | OK |

**Result:** lukealonso 17/19 (89.5%), nvidia 16/19 (84.2%). Key difference: Q9 (algebraic manipulation) — lukealonso correctly computes 20, nvidia incorrectly answers 30.

---

## Conclusions

### 1. MTP does not degrade quality on either engine

Neither SGLang nor vLLM show statistically significant quality loss with MTP enabled. The verification mechanism in speculative decoding guarantees output fidelity. MTP provides 18-24% inference speedup on SGLang — a free speedup.

### 2. Inference engine choice can matter as much as quantization

nvidia NVFP4 on vLLM (88.53%) scores comparably to lukealonso NVFP4 on SGLang (88.28%) on GPQA. nvidia on vLLM GSM8K (98.5%) also outperforms nvidia on SGLang (97.5%). The differences are not statistically significant, but the trend suggests engine-level differences in numerics, scheduling, or attention implementation may influence results.

### 3. lukealonso outperforms nvidia on SGLang

On SGLang, lukealonso/Qwen3.5-397B-A17B-NVFP4 consistently outperforms nvidia/Qwen3.5-397B-A17B-NVFP4 across all benchmarks (+0.8pp to +5.3pp). The advantage is especially pronounced without thinking mode. This aligns with KLD measurements (see [KLD evaluation](kld-evaluation.md)) and community reports (vLLM Issue #36094).

### 4. Recommended production config

```bash
# Model
--model lukealonso/Qwen3.5-397B-A17B-NVFP4

# MTP (speculative decoding) — SGLang
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6

# MTP — vLLM
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# Bug mitigations (SGLang)
--disable-shared-experts-fusion
--disable-custom-all-reduce
```

---

## Raw Data

### GPQA — lukealonso MTP (SGLang)

```json
{
  "scores": ["0.889", "0.879", "0.869", "0.884", "0.904", "0.884", "0.879", "0.874"],
  "mean_score": 0.8828
}
```

### GPQA — lukealonso No MTP (SGLang)

```json
{
  "scores": ["0.864", "0.874", "0.894", "0.869", "0.889", "0.864", "0.874", "0.874"],
  "mean_score": 0.8753
}
```

### GPQA — nvidia MTP (SGLang)

```json
{
  "scores": ["0.859", "0.904", "0.859", "0.884", "0.879", "0.859", "0.874", "0.879"],
  "mean_score": 0.8746
}
```

### GPQA — nvidia No MTP (SGLang)

```json
{
  "scores": ["0.864", "0.859", "0.869", "0.864", "0.848", "0.864", "0.869", "0.889"],
  "mean_score": 0.8658
}
```

### GPQA — nvidia MTP (vLLM)

```json
{
  "scores": ["0.919", "0.879", "0.894", "0.859", "0.864", "0.884", "0.899", "0.884"],
  "mean_score": 0.8853
}
```

### GPQA — nvidia No MTP (vLLM)

```json
{
  "scores": ["0.874", "0.864", "0.869", "0.869", "0.859", "0.859", "0.894", "0.864"],
  "mean_score": 0.8690
}
```
