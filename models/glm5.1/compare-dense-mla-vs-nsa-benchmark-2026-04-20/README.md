# Dense MLA vs NSA vs vLLM Benchmark for GLM-5.1-NVFP4

This directory is a self-contained reproduction bundle for comparing:

- **SGLang dense MLA**
- **SGLang NSA**
- **vLLM** on the currently tested configuration

all on the same `lukealonso/GLM-5.1-NVFP4` long-context task.

## Why this test exists

Luke's working hypothesis is that **dense MLA is not a harmless superset** for a model that was co-trained with an NSA indexer. The argument is that, if the model was always trained with sparse retrieval in the loop, the sparse routing mask is part of the learned architecture rather than an optional optimization. Under that view:

- the indexer shapes which tokens ever compete in attention,
- training gradients are conditioned on that restricted candidate set,
- attention score calibration is learned in the sparse world,
- removing the indexer at inference time changes the function being computed.

That leads to a concrete expectation:

- short contexts may still look mostly fine,
- long contexts may become more distractible under dense inference,
- dense attention may spread probability mass across many distractors,
- some heads may no longer do the job they were trained for,
- quality can fall even when dense attention has access to more tokens.

This benchmark was created to test whether that theoretical claim shows up in a real long-context task.

## Main result

The recommended comparison treats known pathological breakdown runs as exclusions for the SGLang paths:

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

Interpretation:

- `sglang_nsa` gave the best quality result in the filtered comparison.
- `vllm_5248` was second-best on accuracy, but required substantially more completion tokens.
- `sglang_dense_mla` was worst on final-answer accuracy.
- The excluded NSA runs (`21`, `29`) are documented separately as repetition/degeneration bugs and should be read as tail-pathology evidence, not as normal quality data points.

## High-level findings

1. The results do **not** support the idea that dense MLA is a harmless substitute for NSA on this model.
2. `sglang_nsa` had the best quality result and the best token efficiency.
3. `vllm_5248` was competitive on accuracy, but substantially more token-hungry than `sglang_nsa`.
4. `sglang_dense_mla` failed mostly by staying fluent but reasoning to the wrong country.
5. So the comparison is not just about mean accuracy; it is also about **failure shape**:
   - SGLang dense MLA: more coherent-but-wrong
   - SGLang NSA: fewer wrong answers overall, but worse pathological outliers
   - vLLM: coherent wrong answers in this batch, no observed repetition collapse

## Exact runtime state used for the benchmark

Base SGLang source inside the container:

- repo: `lukealonso/sglang`
- commit: `f4b7830ed8d3d570d9662e273009334c824c8227`
- `b12x`: `0.9.6`
- container image used as runtime base: `voipmonitor/sglang:cu130test`

Local modifications present during the benchmark:

- `patches/tokenizer_manager_glm_generate_template.diff`
  - auto-applies the GLM chat template for raw `/generate` requests so that the benchmark does not exercise the wrong prompt contract.
- `patches/nsa_backend_extend_capacity_144k.diff`
  - raises the b12x NSA extend gather workspace cap from `128 * 1024` to `144 * 1024` to avoid a long-context crash.
- `patches/cuda_piecewise_backend_skip_capture.diff`
  - skips piecewise CUDA graph capture for runtime-recompiled shapes when no PCG capture stream is active.
- extra Blackwell tuning JSONs copied under `configs/`
  - these are the local performance tuning configs that were present in the test runtime.

Reference vLLM runtime used for the additional comparison:

- process model: `lukealonso/GLM-5.1-NVFP4`
- served model name: `abtest`
- port: `5248`
- launch script captured in:
  - `scripts/launch_vllm_5248_reference.sh`

## What is included here

```text
compare-dense-mla-vs-nsa-benchmark-2026-04-20/
├── README.md
├── scripts/
│   ├── benchmark_glm_variants.py
│   ├── benchmark_vllm_5248.py
│   ├── launch_dense_mla.sh
│   ├── launch_nsa.sh
│   ├── launch_vllm_5248_reference.sh
│   ├── run_benchmark.sh
│   └── test.py
├── prompts/
│   └── testLuke5.txt
├── patches/
│   ├── tokenizer_manager_glm_generate_template.diff
│   ├── nsa_backend_extend_capacity_144k.diff
│   └── cuda_piecewise_backend_skip_capture.diff
├── configs/
│   ├── quantization/
│   └── moe_triton/
└── results/
    ├── failure_analysis_report.md
    ├── outlier_report_for_luke.md
    ├── summary.json
    ├── engine_comparison.md
    ├── combined_engine_summary.json
    ├── final_summary.csv
    ├── final_summary.json
    ├── wrong_run_classification.json
    ├── dense_mla_runs.jsonl
    ├── nsa_runs.jsonl
    ├── vllm_5248_runs.jsonl
    ├── vllm_5248_final_summary.json
    ├── vllm_5248_wrong_run_classification.json
    ├── datasets/
    │   ├── nsa_without_run21.jsonl
    │   ├── nsa_without_breakdowns.jsonl
    │   └── dense_mla_without_breakdowns.jsonl
    └── full_outputs/
        ├── dense_mla_run_8_full_output.txt
        ├── nsa_run_21_full_output.txt
        ├── nsa_run_29_full_output.txt
        ├── vllm_run_10_full_output.txt
        ├── vllm_run_18_full_output.txt
        ├── vllm_run_24_full_output.txt
        └── vllm_run_25_full_output.txt
```

## How to reproduce the test

### 1. Prepare the runtime

Use the same SGLang source state and apply the local diffs in `patches/`. The benchmark was not run on a fully clean upstream tree.

You should also have the Blackwell tuning JSONs from `configs/` installed into the corresponding SGLang paths:

- quantization configs -> `python/sglang/srt/layers/quantization/configs/`
- MoE triton configs -> `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/`

### 2. Start one of the GLM variants

Dense MLA:

```bash
bash scripts/launch_dense_mla.sh
```

NSA:

```bash
bash scripts/launch_nsa.sh
```

Reference vLLM:

```bash
bash scripts/launch_vllm_5248_reference.sh
```

### 3. Run a single evaluation manually

```bash
python3 scripts/test.py --port 8001 -f prompts/testLuke5.txt --max-tokens 40000
```

### 4. Run the full comparison benchmark

```bash
bash scripts/run_benchmark.sh
```

The benchmark script:

- launches `dense_mla`, runs 30 evaluations, records all outputs,
- stops the server,
- launches `nsa`, runs 30 evaluations, records all outputs,
- scores correctness from the **final answer**, not from intermediate reasoning tokens.

To run the standalone vLLM benchmark:

```bash
python3 scripts/benchmark_vllm_5248.py
```

## How correctness was judged

This is important.

The benchmark does **not** stop as soon as it first sees the token `Estonia` anywhere in the stream. That would be misleading because the model often mentions both candidate countries during reasoning.

Instead, correctness is scored from the **final answer line** emitted at the end of generation.

This makes the benchmark much closer to the real user-facing behavior.

## Failure analysis summary

### dense_mla wrong runs

- coherent but wrong answers: runs `15, 19, 21, 22, 24, 27, 28`
- breakdown / truncated generation: run `8`

Dense MLA therefore mostly failed by producing a fluent but wrong answer rather than by collapsing into repetition.

### nsa wrong runs

- coherent but wrong answers: runs `6, 26, 27`
- repetition / degeneration failures: runs `21, 29`

NSA therefore had a better raw success rate, but a worse tail failure mode.

### vLLM wrong runs

- coherent but wrong answers: runs `10, 18, 24, 25`
- no repetition collapse observed in this 30-run batch

The vLLM failures in this batch were therefore wrong-but-coherent, not obvious generation breakdowns.

## Raw all-run statistics

These numbers are retained for completeness, but they are **not** the recommended comparison because they include the pathological breakdown runs.

```text
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
| Variant   | Completed | Correct | Wrong | Correct rate                  | Completion tokens min/med/avg/max    | Elapsed s min/med/avg/max                 |
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
| dense_mla | 30        | 22      | 8     | 73.33%                        | 3219 / 8681.0 / 8852.767 / 17343     | 36.813 / 101.255 / 106.872 / 204.995      |
| nsa       | 30        | 25      | 5     | 83.33%                        | 1784 / 5051.5 / 7910.567 / 40000     | 25.270 / 77.088 / 164.002 / 1438.703      |
| vllm_5248 | 30        | 26      | 4     | 86.67%                        | 4336 / 10502.0 / 11179.333 / 21555   | 70.236 / 162.094 / 173.836 / 332.113      |
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
```

## Filtered views

### NSA with only the single worst outlier removed (`run 21`)

- completed: 29
- correct: 25
- wrong: 4
- correct rate: 86.21%
- completion tokens min / median / mean / max: 1784 / 5011 / 6804.034 / 40000
- elapsed seconds min / median / mean / max: 25.270 / 76.870 / 143.278 / 1438.703

### NSA with both degeneration runs removed (`runs 21 and 29`)

- completed: 28
- correct: 25
- wrong: 3
- correct rate: 89.29%
- completion tokens min / median / mean / max: 1784 / 4965.5 / 5618.464 / 27297
- elapsed seconds min / median / mean / max: 25.270 / 76.562 / 97.013 / 615.137

## Recommended reading order for Luke

1. Start with this `README.md`.
2. Read `results/engine_comparison.md` for the three-way compare.
3. Read `results/failure_analysis_report.md` for the SGLang dense-vs-NSA breakdown.
4. Read `results/outlier_report_for_luke.md` for the dedicated explanation of the excluded outlier runs.
5. Open `results/wrong_run_classification.json` and `results/vllm_5248_wrong_run_classification.json`.
6. Inspect the raw full-output files for the pathological NSA runs and the wrong vLLM runs if needed.

## Bottom line

This benchmark supports the claim that, for this GLM-5.1 NSA-trained model, **dense MLA is not just a harmless superset of NSA**. On this task, dense MLA was clearly worse than NSA. The additional vLLM result does not reverse that conclusion: it was competitive on accuracy, but far more token-hungry than the SGLang NSA path.

The practical ranking from this bundle is:

- **quality / faithfulness to trained behavior:** `sglang_nsa` best
- **second-best quality:** `vllm_5248`
- **worst quality:** `sglang_dense_mla`
- **token efficiency:** `sglang_nsa` by far the best
