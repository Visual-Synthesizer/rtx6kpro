# Kimi-K3 Full-MXFP4 4096-Token Prefill

This profile serves the official Kimi-K3 MXFP4 target with the Inferact
DSpark draft on TP16/DCP16. It preserves a physical 1,057,049-token FP8 KV
cache while scheduling exact 4,096-token prefill chunks.

Implementation status: **implemented**. Runtime status: **qualified** for the
source composition and machine described below. Deployment status:
**research-only** because the measured container uses read-only source mounts
over a published compiled-package image. A compiled immutable image is not
published for this profile.

The machine-readable receipt is
[`validation/full-mxfp4-p4096-20260819.json`](validation/full-mxfp4-p4096-20260819.json).

## Result

The reference column is the 4,096-token Kimi-K3 profile measured on August 4,
2026. The bounded-workspace column uses the same scheduler size but divides
each scheduler chunk into four 1,024-token MoE launches. The qualified column
uses one 4,096-token B12X MoE launch with bounded route-reduction storage.

| Prompt tokens | August 4 reference | Four MoE launches | 4,096-token bounded reduction | Change from reference | Change from four launches |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 3,450.5 tok/s | 2,723.1 tok/s | 3,861.7 tok/s | +11.92% | +41.81% |
| 32,768 | 3,589.4 tok/s | 2,897.0 tok/s | 3,732.5 tok/s | +3.99% | +28.84% |
| 65,535 | 3,475.2 tok/s | 2,839.7 tok/s | 3,554.4 tok/s | +2.28% | +25.17% |

Each size has one warmup. The 8,192-token result is the median of six measured
runs; the other results are medians of three measured runs. Requests use
direct token IDs, one generated token, streaming time to first token, and a
unique cache salt. Prefix caching and external KV caching are disabled.

The 4,096-token bounded reduction therefore removes the observed 18–21%
regression and exceeds the identified August 4 reference at every measured
length.

## Memory and scheduler contract

| Property | Value |
|---|---:|
| Tensor parallelism | 16 |
| Decode context parallelism | 16 |
| Maximum model length | 1,048,576 tokens |
| Physical KV capacity | 1,057,049 tokens |
| KV allocation | 1,325,000,000 bytes per GPU |
| Scheduler batch-token limit | 4,102 |
| DSpark draft-reserved slots | 6 |
| Effective scheduled-token limit | 4,096 |
| Active sequence limit | 1 |
| B12X MoE workspace token limit | 4,096 |

Speculative scheduling subtracts six draft slots from the batch-token budget
for one active sequence. `MAX_NUM_BATCHED_TOKENS=4102` and
`--max-num-scheduled-tokens 4096` therefore produce an exact 4,096-token
prefill chunk. A batch-token value of 4,096 permits only 4,090 prompt tokens,
which divides an 8,192-token prompt into three scheduler iterations instead of
two.

The opt-in B12X reduction in
[B12X PR #238](https://github.com/local-inference-lab/b12x/pull/238)
writes every FC2 route into a caller-owned per-token FP32 accumulator and casts
to BF16 once after all routes. For Kimi-K3 at 4,096 tokens per rank, total
caller-owned W4A16 scratch is 278.51 MiB: 48 MiB for the FC1 cache, 112 MiB for
the FP32 accumulator, 24 MiB for the activation cache, and 94.51 MiB for GEMM
accumulation scratch and route metadata. Materializing the complete route
output requires 1,014.51 MiB per rank. The bounded path releases 736 MiB per
rank.

## Source composition

The published base image is:

```text
voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa
```

Its four affected vLLM files are byte-identical to
`local-inference-lab/vllm@337ef76dcd30198d8dd47f6c9e61ae1d8be73656`
before applying these independent pull requests:

| Pull request | Runtime role |
|---|---|
| [vLLM #444](https://github.com/local-inference-lab/vllm/pull/444) | Return unoccupied allocator blocks around manual KV profiling and graph capture |
| [vLLM #445](https://github.com/local-inference-lab/vllm/pull/445) | Bound B12X MoE scratch by launch rows and preserve logical output geometry |
| [vLLM #446](https://github.com/local-inference-lab/vllm/pull/446) | Reserve reusable Kimi AttnRes prefill storage before KV allocation |
| [vLLM #447](https://github.com/local-inference-lab/vllm/pull/447) | Reuse packed Kimi vision Q/K buffers during RoPE |

Vision is disabled for the throughput measurement, so vLLM #447 is present in
the exact source composition but does not affect the reported result.

Applying the five listed vLLM commits to revision
`337ef76dcd30198d8dd47f6c9e61ae1d8be73656` produces Git tree
`d4195f16a70f8d441ad5969acb021ee71f39e13b`.

The B12X package is composed from:

```text
master base       c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5
PR #227 head      0eba6ae99e0d1fad6ec268d8c291f498ec1dd4d9
PR #145 head      7f88972df71d580951115220b75923078b769fe8
PR #238 head      450ba32b0580bb35d0dce28f70e1cd5fb7ce8116
code revision     0c3be37138f74a6d0213c10202e0077c2d2a44da
resulting tree    4757592885f764593625d19f99bfba8d7c973b6b
package subtree   1e71cf90fbe4116685819845186439df46722c19
```

The implementation patch ID is
`ac0b86c5901941483654e6e3314f66bb095bf623` in both PR #238 and the B12X
package mounted during runtime qualification. Commits after the code revision
add only checked-in validation receipts.

## Prepare source checkouts

The commands below create the exact file contents mounted by the qualified
container. They do not replace unrelated files in the compiled vLLM package.

```bash
K3_SOURCE_ROOT=/opt/kimi-k3-p4096-source
mkdir -p "${K3_SOURCE_ROOT}"

git clone https://github.com/local-inference-lab/vllm.git \
  "${K3_SOURCE_ROOT}/vllm"
git -C "${K3_SOURCE_ROOT}/vllm" checkout --detach \
  337ef76dcd30198d8dd47f6c9e61ae1d8be73656
git -C "${K3_SOURCE_ROOT}/vllm" fetch \
  https://github.com/voipmonitor/vllm.git \
  fix/ii-manual-kv-profile-cache-release-20260819 \
  fix/ii-b12x-moe-bounded-prefill-20260819 \
  fix/ii-kimi-attnres-prefill-workspace-20260819 \
  fix/ii-kimi-vision-rope-buffer-reuse-20260819
git -C "${K3_SOURCE_ROOT}/vllm" cherry-pick \
  98d19989bb4744814f370dcd76932d02264f6a25 \
  653c7ff980b25f288cebddd02dc92d0fe42f106e \
  d3ab47527c4904c5f9ea510c82236e677facc95a \
  8bdc2ff939616e3603b74dceff3d9088a6a630f6 \
  097ad6ac495490b240aff8356479e5b50d0db78b
git -C "${K3_SOURCE_ROOT}/vllm" rev-parse 'HEAD^{tree}'

git clone https://github.com/local-inference-lab/b12x.git \
  "${K3_SOURCE_ROOT}/b12x"
git -C "${K3_SOURCE_ROOT}/b12x" checkout --detach \
  c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5
git -C "${K3_SOURCE_ROOT}/b12x" fetch origin \
  pull/227/head:refs/remotes/profile/pr-227 \
  pull/145/head:refs/remotes/profile/pr-145 \
  pull/238/head:refs/remotes/profile/pr-238
git -C "${K3_SOURCE_ROOT}/b12x" merge --no-edit --no-ff \
  refs/remotes/profile/pr-227
git -C "${K3_SOURCE_ROOT}/b12x" merge --no-edit --no-ff \
  refs/remotes/profile/pr-145
git -C "${K3_SOURCE_ROOT}/b12x" cherry-pick \
  325528adda0a8f0ec88665dfc86e1f008e9058d5 \
  c3723e707d1afeaea26657c01ab1bbff1c5d71d5 \
  52364a1cae76a89952eb92db4bac57d8457aab62 \
  0c3be37138f74a6d0213c10202e0077c2d2a44da \
  da3fbe34bf9403ba2257122ee001a9b4fb7555bf \
  450ba32b0580bb35d0dce28f70e1cd5fb7ce8116
git -C "${K3_SOURCE_ROOT}/b12x" rev-parse 'HEAD^{tree}'
```

The vLLM tree command must print
`d4195f16a70f8d441ad5969acb021ee71f39e13b`. The B12X tree command must print
`4757592885f764593625d19f99bfba8d7c973b6b`.

## Start the 4,096-token profile

The target and draft snapshots identified below must already exist in the
mounted Hugging Face cache:

```text
moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970
Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6
```

```bash
K3_SOURCE_ROOT=/opt/kimi-k3-p4096-source
K3_CACHE_ROOT=/mnt/luke/kimi-k3-cache/p4096-full-mxfp4
mkdir -p "${K3_CACHE_ROOT}"

docker pull \
  voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa

docker run -d \
  --name kimi-k3-full-mxfp4-p4096 \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -e TP_SIZE=16 \
  -e DCP_SIZE=16 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=1 \
  -e MAX_NUM_BATCHED_TOKENS=4102 \
  -e KV_CACHE_MEMORY_BYTES=1325000000 \
  -e ENABLE_VISION=0 \
  -e ENABLE_PREFIX_CACHING=0 \
  -e LMCACHE_MODE=off \
  -e B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096 \
  -e B12X_W4A16_PREFILL_FUSED_SUM=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e XDG_CACHE_HOME=/cache/jit/p4096 \
  -e B12X_COMPILE_CACHE_DIR=/cache/jit/p4096/b12x/compile \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/p4096/b12x/cute \
  -e CUTE_DSL_CACHE_DIR=/cache/jit/p4096/cute-dsl \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/p4096/torchinductor \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/p4096/torch-extensions \
  -e TRITON_CACHE_DIR=/cache/jit/p4096/triton \
  -e VLLM_CACHE_ROOT=/cache/jit/p4096/vllm \
  -e VLLM_CACHE_DIR=/cache/jit/p4096/vllm \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v "${K3_CACHE_ROOT}":/cache/jit:rw \
  -v "${K3_SOURCE_ROOT}/b12x/b12x":/opt/venv/lib/python3.12/site-packages/b12x:ro \
  -v "${K3_SOURCE_ROOT}/vllm/vllm/model_executor/layers/fused_moe/b12x_moe.py":/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/b12x_moe.py:ro \
  -v "${K3_SOURCE_ROOT}/vllm/vllm/models/kimi_k3/nvidia/model.py":/opt/venv/lib/python3.12/site-packages/vllm/models/kimi_k3/nvidia/model.py:ro \
  -v "${K3_SOURCE_ROOT}/vllm/vllm/model_executor/models/kimi_k25_vit.py":/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/kimi_k25_vit.py:ro \
  -v "${K3_SOURCE_ROOT}/vllm/vllm/v1/worker/gpu_worker.py":/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py:ro \
  voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa \
  --max-num-scheduled-tokens 4096
```

Check model identity and readiness with:

```bash
docker logs -f kimi-k3-full-mxfp4-p4096
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

## Reproduce the prefill measurement

Run the committed benchmark from the wiki checkout:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-prefill.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-MXFP4-DSpark7-DCP16-1M \
  --sizes 8192 32768 65535 \
  --warmups 1 \
  --runs 3 \
  --output /tmp/kimi-k3-full-mxfp4-p4096.json
```

The 8,192-token six-run result was collected with `--sizes 8192 --runs 6`.

## Validation and numerical scope

| Check | Result |
|---|---|
| B12X default W4A16 GPU suite | 292 passed, 16 skipped |
| B12X focused route-reduction group | 5 passed |
| vLLM MoE workspace tests | 4 passed |
| vLLM AttnRes tests | 6 passed |
| vLLM vision-projector tests | 4 passed |
| vLLM manual-KV worker tests | 10 passed |
| Kimi 4,096-token graph replay | completed |
| Physical KV allocation | 1,057,049 tokens |

The direct route reduction changes the order of FP32 additions. The Kimi shape
test measured cosine similarity 1.0 and a maximum BF16 absolute difference of
0.015625 against the materialized-route implementation. Bitwise identity and
bitwise repeat determinism are unsupported. The optimization is opt-in through
`B12X_W4A16_PREFILL_FUSED_SUM=1`.

An A-B-B-A microbenchmark measured 5,214.17 us for materialized reduction and
5,304.32 us for bounded reduction. The bounded kernel is 1.73% slower in
isolation; its end-to-end gain comes from executing one 4,096-token MoE launch
instead of four 1,024-token launches.

The normalized DSpark sanity set measured 31.486 target cycles/s. Emitted
throughput was 109.99 tok/s at 35.62% median draft acceptance. Target cycles
per second, not emitted token rate, is the decode regression metric for this
profile.
