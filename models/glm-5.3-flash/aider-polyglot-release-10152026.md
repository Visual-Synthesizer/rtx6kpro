# GLM-5.3-Flash: aider polyglot benchmark, release-10152026

Measured 2026-10-15. Five complete runs of the
[aider polyglot benchmark](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md)
(225 exercises in C++, Go, Java, JavaScript, Python and Rust) against the
GLM-5.3-Flash checkpoint and image that `PRESET=glm53-tp2` serves by default.

## Summary

| Question | Answer |
|---|---|
| pass@2 (the aider leaderboard number) | **88.0 ± 1.1 %** over five runs (86.7–89.3 %); 95 % CI of the mean 86.6–89.4 % |
| pass@1 | 46.0 ± 1.5 % (44.0–47.6 %) |
| Run-to-run spread | One run moves the result by about ±1.5 points, so a single run cannot resolve differences below 3 points. |
| Exercise agreement | 164 of 225 exercises pass in all five runs, 6 pass in none, 55 depend on the run. |
| Failures caused by the server | None. All 3,322 model requests answered with `finish=stop` except one `length`; no connection errors, one context overflow (rerun). |
| Cost per run | 1.8–2.4 M prompt tokens, 6.4–7.1 M completion tokens, 445–469 s per exercise with reasoning at `max` effort. |

## Setup

| Item | Value |
|---|---|
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026` = `karmic-kraken-20261015.8`, digest `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29` |
| Components | vLLM `8244d1c443`, FlashInfer `6023d031be` (FlashInfer main `5a894aff2` plus the Karmic Kraken CI files), runtime profiles `6a833b698f` |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`, branch `qad2-kda-w8a16-csf`, revision `96f27f8c4e443be5bc486f8a978e3b9e0d3f7125`: step-3171 QAD weights, official BF16 KDA gates / DSA indexer / MLA kv_b, block-FP8 MLA projections and shared experts, MXFP8 KDA q/k/v/o, NVFP4 routed experts with `csf_32x2_v1` scale blocks |
| Preset defaults that matter | `VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill` (BF16 activations for the MXFP8 KDA projections in prefill, W8A8 in decode), `VLLM_GLM53_MTP_DRAFT_HEAD=bf16`, MTP3, routed experts W4A16 in decode and A4 in prefill, `temperature 1.0`, `top_p 0.95`, `reasoning_effort max` |
| Parallelism | Two 16-GPU RTX PRO 6000 Blackwell hosts. Host A: four servers from the preset overridden with `TP=4 DCP=1 MAX_NUM_SEQS=32 KV_CACHE_MEMORY_BYTES=25769803776`. Host B: one server with the same overrides and `--data-parallel-size 4 --data-parallel-mode independent`. An nginx `least_conn` proxy in front of both gave the benchmark one endpoint (256 request slots). |
| Benchmark harness | [Aider-AI/aider](https://github.com/Aider-AI/aider) `5dc9490bb`, [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) `7e0611e`, run through [lavdnone2/aider-bench](https://github.com/lavdnone2/aider-bench) `da5a9c4` (its only local change: the run container's memory limit is an environment variable) |
| aider settings | `edit_format: whole`, `use_repo_map: true`, `max_tokens: 160000`, `timeout: 3600`, `use_temperature: false` (aider sends no temperature, so the server defaults above apply), 32 threads per run, the five runs concurrently |
| Scoring | aider's own test suites per exercise; pass@1 = tests pass on the first reply, pass@2 = tests pass after one round of test feedback |

Each run is a separate `benchmark.py` invocation over all 225 exercises (`run1` … `run5`).

## Results

| run | pass@1 | pass@2 | passed | requests | failed requests | finish=length | malformed replies | context overflow | prompt tokens | completion tokens | s / exercise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1 | 46.2 % | 87.6 % | 197 | 665 | 2 | 0 | 0 | 0 | 2,230,234 | 6,708,817 | 455 |
| run2 | 47.1 % | 88.9 % | 200 | 656 | 5 | 0 | 2 | 0 | 1,771,510 | 6,400,257 | 445 |
| run3 | 44.0 % | 86.7 % | 195 | 672 | 6 | 0 | 0 | 0 | 2,045,816 | 7,086,906 | 469 |
| run4 | 47.6 % | 89.3 % | 201 | 656 | 2 | 0 | 1 | 0 | 2,428,663 | 6,796,840 | 450 |
| run5 | 44.9 % | 87.6 % | 197 | 673 | 6 | 1 | 0 | 1 (rerun) | 2,203,933 | 6,908,890 | 455 |
| **mean ± sd** | **46.0 ± 1.5 %** | **88.0 ± 1.1 %** | 198.0 | | | | | | | | |
| 95 % CI of the mean | 44.1–47.8 % | 86.6–89.4 % | | | | | | | | | |
| min–max | 44.0–47.6 % | 86.7–89.3 % | 195–201 | | | | | | | | |

The standard deviation is over the five runs; the confidence interval uses
Student's t with four degrees of freedom.

### Per language

| language | exercises | pass@1 mean ± sd | pass@2 mean ± sd | pass@2 per run (run1 … run5) |
|---|---|---|---|---|
| cpp | 26 | 39.2 ± 3.2 % | 91.5 ± 4.2 % | 84.6 / 96.2 / 92.3 / 92.3 / 92.3 |
| go | 39 | 62.1 ± 3.3 % | 84.1 ± 3.3 % | 82.1 / 84.6 / 79.5 / 87.2 / 87.2 |
| java | 47 | 36.6 ± 4.6 % | 83.4 ± 4.9 % | 83.0 / 89.4 / 78.7 / 87.2 / 78.7 |
| javascript | 49 | 40.0 ± 1.8 % | 91.4 ± 4.4 % | 93.9 / 83.7 / 93.9 / 93.9 / 91.8 |
| python | 34 | 42.9 ± 3.9 % | 91.8 ± 2.5 % | 94.1 / 94.1 / 91.2 / 88.2 / 91.2 |
| rust | 30 | 58.7 ± 7.7 % | 87.3 ± 1.5 % | 86.7 / 90.0 / 86.7 / 86.7 / 86.7 |

Per-language rates rest on 26–49 exercises, so one exercise is 2–4 points;
the per-run columns show that spread directly.

### Agreement between runs

| exercises passed in k of 5 runs | k=5 | k=4 | k=3 | k=2 | k=1 | k=0 |
|---|---|---|---|---|---|---|
| count | 164 | 25 | 17 | 6 | 7 | 6 |

Never solved in any run: `go/hexadecimal`, `go/matrix`, `go/poker`,
`go/robot-simulator`, `python/dot-dsl`, `rust/dot-dsl`.

### Anomalies

| Item | Count | Effect on the score |
|---|---|---|
| Exercises with an error output (run2: `cpp/circular-buffer`, `cpp/zebra-puzzle`; run4: `cpp/zebra-puzzle`; run5: `javascript/ocr-numbers`) | 4 | None: all four passed on the second try. |
| Context overflow (`javascript/ocr-numbers`, run5) | 1 | Set aside and rerun with `--redo exhausted`; the rerun passed. |
| Reply cut at `max_tokens` (`finish=length`, run5) | 1 | Counted as is. |
| Test timeout (`go/robot-simulator`, run3) | 1 | Counted as failed; this exercise failed in every run. |
| Failed requests (`RuntimeError: cannot schedule new futures after shutdown`) | 21 over five runs | Raised by aider's own thread pool while a run shuts down, not by the server; no exercise lost a try to them. |

### Context

A community measurement posted in the Local Inference Lab Discord on
2026-10-14 reported pass@2 89.3 % for a different NVFP4 checkpoint of the same
model served by SGLang, from one run. That value lies inside the run-to-run
spread measured here, so the two serving stacks are not distinguishable by a
single run of this benchmark.

## Decode throughput of the serving topologies

Measured on one 16-GPU host with `llm_decode_bench` from the image against
every server at once (30 s per cell, context 0, MTP3; client-measured tokens
per second summed over the servers). The preset's TP2 recipe is the qualified
default; the TP4 variants were used for this benchmark because they serve the
same checkpoint with four times the request slots per GPU pair.

| Topology on 16 GPUs | Concurrent requests | tok/s |
|---|---|---|
| 8 × TP2/DCP2 (preset as shipped, 8 slots each) | 64 | 4,953 |
| 1 × TP2 with `--data-parallel-size 8 --data-parallel-mode independent` | 64 | 4,708 |
| 4 × TP4/DCP1, 32 slots each | 128 | 7,650 |
| 4 × TP4/DCP1, 32 slots each | 256 | 7,299 |
| 1 × TP4 with `--data-parallel-size 4 --data-parallel-mode independent` (host B) | 128 | 8,858 |
| 1 × TP4 with `--data-parallel-size 4 --data-parallel-mode independent` (host B) | 256 | 7,815 |

Data-parallel mode and separate servers give the same throughput within the
host-to-host difference; data-parallel mode needs no external load balancer.

## Reproduction

```bash
# one TP4/DCP1 server (four GPUs); repeat per GPU group with its own PORT, 100 apart
docker run -d --name glm-tp4 --init --gpus '"device=0,1,2,3"' --restart on-failure \
  --network host --ipc host --shm-size 32g --stop-timeout 30 \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v model-cache:/root/.cache/huggingface -v glm-tp4-runtime:/cache \
  -e PRESET=glm53-tp2 -e TP=4 -e DCP=1 -e MAX_NUM_SEQS=32 -e KV_CACHE_MEMORY_BYTES=25769803776 -e PORT=9200 \
  ghcr.io/local-inference-lab/vllm:release-10152026

# or all 16 GPUs behind one port
docker run -d --name glm-dp4 --init --gpus all --restart on-failure \
  --network host --ipc host --shm-size 32g --stop-timeout 30 \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v model-cache:/root/.cache/huggingface -v glm-dp4-runtime:/cache \
  -e PRESET=glm53-tp2 -e TP=4 -e DCP=1 -e MAX_NUM_SEQS=32 -e KV_CACHE_MEMORY_BYTES=25769803776 -e PORT=9200 \
  ghcr.io/local-inference-lab/vllm:release-10152026 -- --data-parallel-size 4 --data-parallel-mode independent

# one benchmark run (lavdnone2/aider-bench; the URL must be reachable from the benchmark's Docker container)
./aider-bench.sh --run --url http://<host>:9200/v1 --model GLM-5.3-Flash --format whole \
  --max-tokens 160000 --threads 32 --request-timeout 3600 --stats run1 --use_temperature false
./aider-bench.sh --run --stats run1 --redo exhausted   # rerun context overflows, if any
./aider-bench.sh --show --stats run1
```

## Data

[`aider-polyglot-release-10152026/`](aider-polyglot-release-10152026/): the
aggregate (`aggregate.md`, `aggregate.json`, produced by `aggregate.py`), and
per run the aider statistics (`runN-stats.yml`), one line per exercise
(`runN-results.jsonl`: pass, try, duration, tokens, anomaly counters) and one
line per model request (`runN-requests.jsonl`: parameters sent, finish reason,
tokens, errors).
