# Kimi-K3 QSRT K2 TP16/DCP8 Serving

## Purpose

This page specifies target-only and Inferact DSpark serving for
`lukealonso/Kimi-K3-QSRT-K2` on 16 NVIDIA RTX PRO 6000 Blackwell Workstation
Edition GPUs. The checkpoint stores all 82,432 routed experts in QSRT K2 and
stores nonexpert linear tensors in MXFP8. Its canonical expert payload is
tensor-parallel independent.

Status: **implemented** for TP16/DCP8 target-only and DSpark serving. The
source-locked image passes source, import, and CPU regression checks. The
source-locked runtime receipt qualifies QSRT K2 target-only decode on TP8/DCP8
with one active sequence; it does not qualify TP16/DCP8 throughput.

## Artifact identities

| Object | Durable identifier |
|---|---|
| Target checkpoint | `lukealonso/Kimi-K3-QSRT-K2@3b98114115f1d41ce7963ba346c3fca19918b0bd` |
| Official weight source | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| DSpark checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| Docker image | `voipmonitor/vllm:kimi-k3-ii-vllmf21a391-b12x3be8bc7-cu133-torch213-20260816-r2` |
| Registry digest | `sha256:9230c19c6b16ca6216613360619b0cca2356dba65c2297c99817750b3f9e4b83` |
| Docker recipe | [`blackwell-llm-docker@2c469ba`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/2c469ba2c54827d82b96b57450374b9c46f163ac) |
| Runtime base image | `voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971` |
| vLLM review units | [#382](https://github.com/local-inference-lab/vllm/pull/382) through [#391](https://github.com/local-inference-lab/vllm/pull/391) |
| vLLM resulting tree | `f21a391de0a1b127c93ac718fd7d1818f025317b` |
| B12X review units | [#215](https://github.com/local-inference-lab/b12x/pull/215) and [#220](https://github.com/local-inference-lab/b12x/pull/220) |
| B12X resulting tree | `3be8bc74d6813223b5be732a4c865401a693f5f5` |

The image uses stable PyTorch 2.13.0, CUDA 13.3, patched NCCL 2.31.2,
InstantTensor 0.1.9, FlashInfer 0.6.18+cu133, and XGrammar 0.2.5. The image source
composition applies hash-verified minimal review units to immutable vLLM and
B12X base commits and verifies each resulting Git tree during the build.

## Serving profiles

Both profiles use TP16/DCP8, B12X dense MLA, B12X routed MoE and linear
kernels, B12X DCP all-to-all, FP8 target KV cache, Triton KDA prefill,
FlashAttention 2 MLA prefill, and InstantTensor checkpoint loading. Prefix
caching is disabled.

| Property | Target-only | DSpark K7 |
|---|---:|---:|
| Target KV allocation per rank | 1,950,000,000 bytes | 2,300,000,000 bytes |
| Maximum model length | 1,048,576 tokens | 1,048,576 tokens |
| Scheduler token budget | 4,096 | 4,096 |
| Maximum active sequences | 8 | 8 |
| CUDA graph batch sizes | 1, 8 | 8, 16, 24, 32, 40, 48, 56, 64 |
| Speculative tokens | disabled | 7 |
| Draft KV window | n/a | 32,768 tokens |
| Draft linear format | n/a | online MXFP8 except `fused_qkv_a_proj` in BF16 |

The DSpark draft transformer is replicated across tensor-parallel ranks. Its
Markov head uses the vLLM sharded execution path while the first Markov matrix
remains replicated. The target and draft use independent FP8 KV caches.

TP16/DCP4 is implemented and numerically qualified for target-only execution,
but its measured 20.29 tok/s rate under a 256-token prompt and 384-token output
does not qualify it as the serving default. The exact runtime and measurement
are stored in
[`validation/qsrt-k2-tp16-dcp4-20260814.json`](validation/qsrt-k2-tp16-dcp4-20260814.json).

## Cache isolation

Use a separate writable JIT cache for each profile. The profiles compile
different graph shapes and generated kernels.

```bash
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp16-nospec
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp16-dspark
```

The commands below expose the API directly on host port 8001. Replace
`PORT=8001` with `PORT=8000` when port 8000 is available.

## Start target-only serving

```bash
docker run -d \
  --name kimi-k3-qsrt-nospec \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp16-nospec:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-qsrt-nospec \
  voipmonitor/vllm@sha256:9230c19c6b16ca6216613360619b0cca2356dba65c2297c99817750b3f9e4b83
```

The served model name is `Kimi-K3-QSRT-K2-NoSpec-TP16-DCP8-1M`.

## Start DSpark serving

```bash
docker run -d \
  --name kimi-k3-qsrt-dspark \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp16-dspark:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-qsrt-dspark \
  voipmonitor/vllm@sha256:9230c19c6b16ca6216613360619b0cca2356dba65c2297c99817750b3f9e4b83
```

The served model name is `Kimi-K3-QSRT-K2-DSpark7-TP16-DCP8-1M`.

## Readiness and API smoke test

Wait for both the health endpoint and the requested model identity:

```bash
until curl -fsS http://127.0.0.1:8001/health >/dev/null; do sleep 5; done

curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K3-QSRT-K2-DSpark7-TP16-DCP8-1M",
    "messages": [{"role": "user", "content": "Reply with READY."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

The server log must report at least 1,048,576 physical target KV tokens before
the profile is described as a one-million-token runtime.

## Fixed-token decode measurement

The repository stores one immutable 256-token prompt in
`models/kimi-k3/tools/decode-baseline-256-token-ids.json`. The token file has
SHA-256
`64c2ff11a83dda71caea12ea10b16511b0114c15973b949cde3ebd2bc4c712af`.

Target-only measurement constrains every generated token to token ID 13. This
removes output-dependent sampling and sequence-length variation:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-nospec-decode.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-QSRT-K2-NoSpec-TP16-DCP8-1M \
  --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
  --prompt-tokens 256 \
  --max-tokens 1024 \
  --warmups 2 \
  --runs 8 \
  --output-dir /tmp/kimi-k3-qsrt-nospec-normalized
```

DSpark measurement uses greedy sampling with seed 1 and `ignore_eos=true`. It
records output rate, accepted draft-token fraction, emitted tokens per target
cycle, and target cycles per second:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-dspark-decode.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-QSRT-K2-DSpark7-TP16-DCP8-1M \
  --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
  --prompt-tokens 256 \
  --max-tokens 1024 \
  --warmups 1 \
  --runs 8 \
  --output-dir /tmp/kimi-k3-qsrt-dspark-normalized
```

The two protocols answer different questions. The target-only constrained-token
rate isolates runtime decode cost. The DSpark rate includes output-dependent
draft acceptance and must be reported with its acceptance and target-cycle
statistics.

## Build the image

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 2c469ba2c54827d82b96b57450374b9c46f163ac

RELEASE_DATE=20260816 REVISION=r2 ./build-kimi-k3-qsrt-tp16-runtime.sh
```

The builder verifies the base-image identity, integration patch hashes,
resulting vLLM and B12X Git trees, imported source locations, launch-script
syntax, and the FlashAttention CuTeDSL wrapper hash.

## Evaluation boundaries

AA-LCR qualifies the complete checkpoint-and-serving configuration. It keeps
the dataset, prompt, tokenizer, sampling, repeats, and equality checker fixed,
but allows runtime topology and kernels to match each checkpoint. See
[`aa-lcr-reproduction.md`](aa-lcr-reproduction.md).

The distribution-fidelity suite isolates checkpoint changes by replaying one
canonical LM head over runtime-matched reference and candidate hidden states.
The routed-experts-only comparison combines QSRT K2 experts with official BF16
nonexpert tensors so the measured KLD excludes the checkpoint's MXFP8 dense
overlay. See
[`distribution-fidelity-1024x2048.md`](distribution-fidelity-1024x2048.md).
