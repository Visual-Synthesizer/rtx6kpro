# Qwen3.8-Flash-Next

Serve `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` using the
`qwen38-flash-next` profile in the [shared Docker guide](../docs/unified-vllm-docker.md).
The image and launcher are shared with GLM and DeepSeek; no GLM entrypoint
bypass or copied kernel environment is needed. This is not
[Qwen3.8-27B](qwen38-27b.md).

Status: **implemented** TP1/TP2 profile, with **qualified** bounded TP1/MTP3
text measurements below. TP2, vision and no-MTP performance are not part of
that wheel-image qualification. The measured prefill difference remains open.

## Start on one GPU: TP1

Select `LIL_IMAGE` in the [shared image section](../docs/unified-vllm-docker.md#select-the-image),
then use these values with the [common launch command](../docs/unified-vllm-docker.md#start-a-server):

```bash
PROFILE=qwen38-flash-next
GPU_DEVICES=0
TP=1
PORT=8000
SERVE_ARGS=(--mode mtp --draft-tokens 3)
```

TP1 means one 96-GB GPU. The API model name is `Qwen3.8-Flash-Next`.
The named Hugging Face volume downloads/reuses the model by repository name.
The shared command leaves GPU clocks unchanged.

For no speculation, replace the argument array with `SERVE_ARGS=(--mode off)`.
Do not set positive draft-token counts together with `--mode off`.

## Start on two GPUs: TP2

Use a different available pair, and stop an overlapping instance before reuse:

```bash
PROFILE=qwen38-flash-next
GPU_DEVICES=0,1
TP=2
PORT=8000
SERVE_ARGS=(--mode mtp --draft-tokens 3)
```

This changes both visible devices and tensor parallelism. It does not disable
PLE offload or change the checkpoint. TP2 is **implemented**, not newly timed
by the TP1 comparison. B12X/NCCL collectives apply to TP2; TP1 performs no
multi-GPU all-reduce.

The optional [Compose example](qwen38-flash-next/qwen38-flash-next.compose.yml)
uses the same profile interface and requires `LIL_IMAGE` from the shared guide:

```bash
curl -fLO https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml
GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
```

Select `--profile tp2` and `GPU0`/`GPU1` for two GPUs. Do not enable both Compose
profiles on the same port. The Compose file contains deployment settings,
not a second copy of B12X or scheduler policy.

## Precision, model tables and cache

| Setting | Profile behavior |
|---|---|
| Weight format | Mixed ModelOpt NVFP4; the repository name does not mean every tensor is four-bit |
| PLE n-gram tables | CPU-mapped host RAM, `VLLM_PLE_CPU_OFFLOAD=1` |
| Vocabulary projections | BF16 target; private NVFP4 MTP head with BF16 activations |
| Kernels | B12X MoE, dense and GDN decode; native Qwen attention selection |
| KV / recurrent state | FP8 attention KV; recurrent state follows the native model contract |
| Runner / graphs | V2, full-and-piecewise, graph cap 64 |
| Scheduler / context | 6019 tokens, 16 sequences, maximum context 262,144 |
| Prefix cache | Enabled; native `auto` selects exact recurrent request boundaries where supported |
| Vision | Text-only by default; `--no-language-model-only` enables the model path |
| LMCache | Unsupported by this profile; not enabled by PLE offload |

PLE is a learned embedding table, not request KV and not an n-gram speculator.
Historical startup accounting records about 26.82 GiB of mapped host tables;
leave additional host RAM for loading and the server. Keep offload enabled for
the one-96-GB-GPU recipe. A device-resident PLE alternative requires a separate
memory and correctness qualification.

The shared guide explains [prefix retention](../docs/unified-vllm-docker.md#prefix-cache-defaults).
Do not add a global `--prefix-cache-retention-interval 4096` override.
For vision, append `--no-language-model-only` to the argument array. The flag
is implemented, but the text-only measurements below do not qualify vision
correctness or throughput on the published beta digest.

Request sampling in the recorded tests is temperature 1/top-p .95/top-k 20.
The profile leaves checkpoint generation configuration authoritative rather
than claiming all checkpoint revisions have identical server defaults.
For a non-thinking request, use
`"chat_template_kwargs":{"enable_thinking":false}`; it is a different
workload from a reasoning benchmark.

## Measured performance

Stock RTX PRO 6000 Workstation, TP1/MTP3, CPU PLE, FP8 KV, 6019-token budget,
16 sequences, explicit eight-GiB KV allocation, context-zero decode and
temperature 1/top-p .95/top-k 20. Three warmed 30-second decode runs; uncached
nominal-32K prefill uses client time to first token. C8 is aggregate.

| Metric | Community R35 → wheel image | Change |
|---|---:|---:|
| C1 output | 172.92 → 190.11 tok/s | +9.94% |
| C8 output | 664.89 → 689.93 tok/s | +3.77% |
| 32K prefill, one sustained window | 15,387 → 15,162 tok/s | −1.46% |
| C1 request-verifier rate | 81.90 → 83.47 steps/s | +1.92% |

All four API checks and six decode cells pass. Acceptance changes from 2.105
to 2.280, so the C1 output gain is not an isolated kernel speedup.
[Exact image boundary, commands and raw samples](../benchmarks/prepared-b12x-serving/).

A separate three-window prefill repeat on the indexed-PCIe-plan integration
image measures R35 **15,258** versus **15,031 tok/s**, a **−1.49%** median
difference. The windows are independent warmed measurements within one
startup per image, not three independent startups. The small gap remains
**unresolved**, without assigning it to a particular kernel or PR.
[Repeat conditions and receipts](../benchmarks/prepared-b12x-contracts/#qwen-prefill-repeat).

The eight-GiB comparison has 517,581 logical KV tokens. A separate automatic
KV-sizing startup passes graph capture and API checks with 773,216 tokens;
it is not another throughput measurement. These are shared-pool capacities,
not the context limit of an individual request. Use engine-reported logical
capacity rather than physical-block count times page size.

## Quality evaluation and historical releases

- [Published NVFP4 versus QAD AA-LCR](qwen38-flash-next/aa-lcr-nvfp4-vs-qad.md).
- [Direct-answer arithmetic stability](qwen38-flash-next/direct-arithmetic-stability-nvfp4-vs-qad.md).
- [Community R35 deployment and measurement archive](qwen38-flash-next-community-r35.md):
  +6000-clock results, SGLang/Sieve comparisons, older capacity measurements
  and exact recipe boundaries. These are not measurements of the wheel image.

No Qwen Sieve rerun or TP2 speed is claimed for the wheel comparison.
Source review and unresolved items: [issue #773](https://github.com/local-inference-lab/vllm/issues/773).
