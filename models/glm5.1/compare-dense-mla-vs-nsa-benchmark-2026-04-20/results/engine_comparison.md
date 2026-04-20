# Engine Comparison: SGLang Dense MLA vs SGLang NSA vs vLLM

This file compares the three engine/configuration families that were actually benchmarked on the same `GLM-5.1-NVFP4` long-context task.

## Recommended comparison

The recommended top-line comparison treats known pathological breakdown runs as exclusions for the SGLang runs:

- `sglang dense_mla`: exclude run `8`
- `sglang nsa`: exclude runs `21`, `29`
- `vllm_5248`: no exclusions in this 30-run batch

| Engine | Path | Excluded runs | Completed | Correct | Wrong | Correct rate |
|---|---|---|---:|---:|---:|---:|
| `sglang_dense_mla` | dense MLA | `8` | 29 | 22 | 7 | 75.86% |
| `sglang_nsa` | NSA | `21, 29` | 28 | 25 | 3 | 89.29% |
| `vllm_5248` | vLLM current config | none | 30 | 26 | 4 | 86.67% |

| Engine | Median completion tokens | Mean completion tokens | Median elapsed s | Mean elapsed s | Median gen tok/s | Mean gen tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `sglang_dense_mla` | 8631.0 | 8560.000 | 100.648 | 103.488 | 86.621 | 85.731 |
| `sglang_nsa` | 4965.5 | 5618.464 | 76.562 | 97.013 | 68.029 | 67.085 |
| `vllm_5248` | 10502.0 | 11179.333 | 162.094 | 173.836 | 64.373 | 64.466 |

## Main takeaways

1. `sglang_nsa` had the best quality result in the filtered comparison.
2. `vllm_5248` was second-best on accuracy, and in this 30-run batch it did not show the repetition-collapse failures seen in the raw SGLang NSA runs.
3. `sglang_dense_mla` was clearly worst on final-answer accuracy.
4. `sglang_nsa` was also by far the most token-efficient of the three engines.
5. `vllm_5248` used substantially more completion tokens than either SGLang path.

## Failure-shape comparison

- `sglang_dense_mla`
  - 7 coherent wrong answers after excluding the single breakdown run
  - typical wrong answer: `Latvia`
- `sglang_nsa`
  - 3 coherent wrong answers after excluding the two pathological breakdown runs
  - raw all-run view also contains two repetition/degeneration failures
- `vllm_5248`
  - 4 wrong answers in 30 runs
  - all 4 were coherent wrong answers ending in `Latvia`
  - no repetition collapse was observed in this batch

## Interpretation

The comparison still supports Luke's theoretical point: dense MLA underperformed NSA on this NSA-trained model. The strongest evidence is that `sglang_nsa` beat `sglang_dense_mla` both on final-answer accuracy and on token efficiency.

The vLLM result adds an additional data point:

- it does not rescue the dense-MLA hypothesis,
- because it still does not outperform the filtered SGLang NSA result,
- and it reaches its answers with much longer completions.

So the current picture is:

- dense MLA looks worst for quality,
- NSA looks best for quality and token efficiency,
- vLLM looks competitive on accuracy but considerably more token-hungry.

## Files relevant to vLLM

- `results/vllm_5248_runs.jsonl`
- `results/vllm_5248_final_summary.json`
- `results/vllm_5248_wrong_run_classification.json`
- `results/full_outputs/vllm_run_10_full_output.txt`
- `results/full_outputs/vllm_run_18_full_output.txt`
- `results/full_outputs/vllm_run_24_full_output.txt`
- `results/full_outputs/vllm_run_25_full_output.txt`
- `scripts/benchmark_vllm_5248.py`
- `scripts/launch_vllm_5248_reference.sh`
