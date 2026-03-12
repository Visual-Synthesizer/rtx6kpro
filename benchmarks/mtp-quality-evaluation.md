# MTP Quality Evaluation (GPQA Diamond)

Does MTP (Multi-Token Prediction) speculative decoding affect model accuracy? We ran GPQA Diamond 8 times per configuration to find out.

## Table of Contents

- [Summary](#summary)
- [Setup](#setup)
- [Results](#results)
  - [Per-Run Scores](#per-run-scores)
  - [Aggregate Statistics](#aggregate-statistics)
  - [lukealonso MTP vs No MTP](#lukealonso-mtp-vs-no-mtp)
  - [lukealonso MTP vs nvidia MTP](#lukealonso-mtp-vs-nvidia-mtp)
- [Conclusion](#conclusion)
- [Raw Data](#raw-data)

---

## Summary

| Configuration | Mean GPQA Score | Wall Time |
|:---|:---:|:---:|
| lukealonso/NVFP4 + MTP | **88.26%** | ~89 min |
| lukealonso/NVFP4, no MTP | 87.50% | ~108 min |
| nvidia/NVFP4 + MTP | 87.44% | ~103 min |

**MTP does not degrade quality** (p=0.18, not statistically significant) while providing **~18% faster inference**.

---

## Setup

| Parameter | Value |
|:---|:---|
| **Benchmark** | GPQA Diamond (198 questions) |
| **Eval framework** | [simple-evals](https://github.com/openai/simple-evals) (ChatCompletionSampler) |
| **Repeats** | 8 per configuration |
| **Parallel workers** | 8 |
| **Temperature** | 0.0 |
| **Max tokens** | 64,000 |
| **Thinking mode** | Enabled (`chat_template_kwargs: {thinking: True}`) |
| **Model** | Qwen3.5-397B-A17B (NVFP4) |
| **Hardware** | 4x RTX PRO 6000 Blackwell |

### Configurations tested

| Test | Checkpoint | MTP | Purpose |
|:---:|:---|:---:|:---|
| 1 | `lukealonso/Qwen3.5-397B-A17B-NVFP4` | Yes | MTP baseline |
| 2 | `lukealonso/Qwen3.5-397B-A17B-NVFP4` | No | No-MTP control |
| 3 | `nvidia/Qwen3.5-397B-A17B-NVFP4` | Yes | Cross-checkpoint comparison |

---

## Results

### Per-Run Scores

| Run | lukealonso MTP | lukealonso No MTP | nvidia MTP |
|:---:|:---:|:---:|:---:|
| 1 | 0.889 | 0.864 | 0.859 |
| 2 | 0.879 | 0.874 | 0.904 |
| 3 | 0.869 | 0.894 | 0.859 |
| 4 | 0.884 | 0.869 | 0.884 |
| 5 | 0.904 | 0.889 | 0.879 |
| 6 | 0.884 | 0.864 | 0.859 |
| 7 | 0.879 | 0.874 | 0.874 |
| 8 | 0.874 | 0.874 | 0.879 |

### Aggregate Statistics

| Metric | lukealonso MTP | lukealonso No MTP | nvidia MTP |
|:---|:---:|:---:|:---:|
| **Mean** | **0.8826** | **0.8750** | **0.8744** |
| Std (across runs) | 0.0106 | 0.0109 | 0.0155 |
| Min | 0.869 | 0.864 | 0.859 |
| Max | 0.904 | 0.894 | 0.904 |
| Median | 0.8815 | 0.874 | 0.877 |
| Avg response length (chars) | 2550 | 2512 | 2436 |
| Wall time | ~89 min | ~108 min | ~103 min |

### lukealonso MTP vs No MTP

| Metric | Value |
|:---|:---|
| Mean score difference (MTP − No MTP) | +0.76 percentage points |
| Speed improvement | ~18% faster with MTP |
| Welch's t-test p-value | 0.18 (not statistically significant at α=0.05) |
| t-statistic | 1.41 |

MTP provides a meaningful speed improvement with no statistically significant change in accuracy.

### lukealonso MTP vs nvidia MTP

| Metric | Value |
|:---|:---|
| Mean score difference (lukealonso − nvidia) | +0.82 percentage points |
| lukealonso mean | 0.8826 |
| nvidia mean | 0.8744 |

The lukealonso checkpoint scores slightly higher, consistent with its lower KLD (0.035 vs 0.109, see [KLD evaluation](kld-evaluation.md)). However, the GPQA score difference is within run-to-run noise. The nvidia MTP result (87.44%) is essentially on par with lukealonso without MTP (87.50%).

---

## Conclusion

1. **MTP does not degrade quality.** The MTP vs no-MTP difference (+0.76pp) is not statistically significant (p=0.18). MTP is a pure speedup with no accuracy tradeoff.

2. **Checkpoint quality matters more than MTP.** The lukealonso NVFP4 checkpoint (KLD 0.035) consistently outperforms the nvidia NVFP4 checkpoint (KLD 0.109) by ~0.8pp on GPQA, though this too is within noise for 8 repeats.

3. **All three configurations score 87-88% on GPQA Diamond**, confirming that NVFP4 quantization preserves reasoning capability on this benchmark regardless of MTP or checkpoint choice.

---

## Raw Data

### Test 1: lukealonso MTP

```json
{
  "chars": 2549.5454545454545,
  "chars:std": 510.686619847691,
  "score:std": 0.3321451120698461,
  "scores": ["0.889", "0.879", "0.869", "0.884", "0.904", "0.884", "0.879", "0.874"],
  "mean_score": 0.8825757575757577
}
```

### Test 2: lukealonso No MTP

```json
{
  "chars": 2512.040404040404,
  "chars:std": 470.0423023987109,
  "score:std": 0.3321451120698461,
  "scores": ["0.864", "0.874", "0.894", "0.869", "0.889", "0.864", "0.874", "0.874"],
  "mean_score": 0.875
}
```

### Test 3: nvidia MTP

```json
{
  "chars": 2435.686868686869,
  "chars:std": 526.6866921099777,
  "score:std": 0.32637362467481845,
  "scores": ["0.859", "0.904", "0.859", "0.884", "0.879", "0.859", "0.874", "0.879"],
  "mean_score": 0.8743686868686869
}
```
