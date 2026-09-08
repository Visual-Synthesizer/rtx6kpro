# Qwen3.8-Flash-Next

Run [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
with the Jovian Judgement community image. Tensor Parallelism (TP) of one means
**one GPU**, not four. The qualified text-serving configuration uses three-token
Multi-Token Prediction (MTP), abbreviated MTP3 here, and offloads the per-layer
n-gram embedding (PLE) table to host RAM. It is a different model from
[Qwen3.8-27B](qwen38-27b.md).

```text
voipmonitor/vllm:jovian-judgement-community-20260908-r28.1
```

The image contains the same vLLM/B12X runtime as [GLM-5.3-Flash](glm-5.3-flash.md),
but Qwen needs its own launch arguments. The Compose recipe below bypasses the
image's GLM entrypoint. No source mounts or absolute checkpoint paths are needed.

## Start on one GPU: TP1

Status: **qualified** for text, MTP3, 8-bit floating-point (FP8) key/value (KV)
cache, GPU prefix reuse and the performance measurements below on one 96 GB RTX PRO 6000 Blackwell
Workstation Edition GPU. The recipe does not change GPU clocks.

Download the [Compose file](qwen38-flash-next/qwen38-flash-next.compose.yml):

```bash
curl -fL -o qwen38-flash-next.compose.yml \
  https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/qwen38-flash-next/qwen38-flash-next.compose.yml

GPU=0 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp1 up -d
docker compose -f qwen38-flash-next.compose.yml --profile tp1 logs -f qwen-tp1
```

`GPU` selects the physical card for TP1; `GPU0` and `GPU1` select the pair for
TP2. Choose unused GPUs and a port. Requirements are Linux, Docker Compose with
GPU reservations/profiles, NVIDIA Container Toolkit and a driver compatible
with the image's CUDA 13.3 runtime. Model weights occupy approximately 98.5 GB
on disk, in addition to the Docker image and compiler cache. A first launch
downloads the model by repository name; later launches reuse named volumes.
Model loading and CUDA graph capture take several minutes even with local weights.

The API model name is **`Qwen3.8-Flash-Next`**:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Explain what a hash table is."}],"temperature":1,"top_p":0.95,"top_k":20,"max_tokens":2048}'
```

Use `http://SERVER:8000/v1` in an OpenAI-compatible client. The recipe binds all
host interfaces without authentication; expose it only on a trusted network
or behind an authenticated proxy. For a non-thinking request, add
`"chat_template_kwargs":{"enable_thinking":false}` to the JSON body.
That is a different workload from the reasoning benchmark below.

## Start on two GPUs: TP2

Status: **implemented**, with a statically checked recipe; TP2 serving and
performance have **not been qualified on this R28.1 image**. Measurements from
other Qwen-specific images are not substituted for that missing result.

Select two distinct available GPUs. Stop the TP1 service before switching
profiles, because both profiles use the same API port:

```bash
docker compose -f qwen38-flash-next.compose.yml --profile tp1 down
GPU0=0 GPU1=1 PORT=8000 docker compose -f qwen38-flash-next.compose.yml --profile tp2 up -d
docker compose -f qwen38-flash-next.compose.yml --profile tp2 logs -f qwen-tp2
```

The TP2 profile changes both the exposed device pair and
`--tensor-parallel-size 2`. MTP3, PLE offload, precision, token budget and graph
settings remain identical. B12X PCIe all-reduce is enabled for eligible
collectives, with the NVIDIA Collective Communications Library (NCCL)
available for other sizes; this requires working GPU peer access. At TP1 no
multi-GPU collective is executed.

Do not activate both profiles together. The named model/cache volumes survive
`down`; do not add `--volumes` unless deleting those caches is intended.

## N-gram / PLE offload

**`VLLM_PLE_CPU_OFFLOAD=1` is enabled in both recipes.** It puts the model's
large n-gram lookup table in CUDA-mapped host RAM. GPU kernels read the required
entries over PCIe; the rest of the model still runs on the selected GPUs.
Offloading changes storage placement, not the checkpoint's quantization.
It is neither LMCache nor n-gram speculative decoding.

The qualified TP1 startup records **26.82 GiB of mapped host RAM for PLE** and
**74.64 GiB of GPU model-loading memory**. Host RAM is also needed for the server
and loading buffers; 26.82 GiB is not a complete host-memory requirement.
Keep offload enabled for this one-96-GB-GPU recipe. Moving that table back into
GPU memory would exceed its available budget before allocating KV cache.

With two GPUs and sufficient memory, device-resident PLE can be selected for a
separate experiment:

```bash
VLLM_PLE_CPU_OFFLOAD=0 GPU0=0 GPU1=1 PORT=8000 \
  docker compose -f qwen38-flash-next.compose.yml --profile tp2 up -d
```

Status of device-resident PLE on R28.1: **implemented, not qualified**. No
offload-on/off speed comparison is claimed here. Changing this environment
setting recreates the serving container; use it only in a maintenance window.
The native override `--additional-config '{"ple_table_memory":"mapped_host"}'`
or `'{"ple_table_memory":"device"}'` takes precedence over the environment
setting when supplied to vLLM.

## Precision, backends and cache

The checkpoint combines NVIDIA 4-bit floating-point weights (NVFP4) with
other precisions. BF16 denotes bfloat16; W4A16 denotes 4-bit weights with
16-bit activations. Qwen sparse attention is abbreviated QSA. The precision
switches below control the vocabulary projection, not the entire model.

| Component | Recipe setting |
|---|---|
| Checkpoint | Mixed ModelOpt quantization; the NVFP4 name does not mean every tensor uses 4-bit precision |
| Target vocabulary head | BF16, preserved by `VLLM_MXFP8_LM_HEAD=0` |
| MTP draft vocabulary head | Private NVFP4 weight copy, selected by `VLLM_MTP_NVFP4_LM_HEAD=1` |
| Draft-head activations | BF16, enforced by `VLLM_LM_HEAD_A16=1`: W4A16, not W4A4 |
| Linear layers / mixture-of-experts layers / recurrent decode | B12X |
| Qwen sparse attention | Complete B12X QSA operation |
| Gated DeltaNet recurrent prefill | FlashInfer, not GLM's FlashKDA backend |
| Attention KV / recurrent state | FP8 / FP32 |
| Model runner / CUDA graphs | V2 / `FULL_AND_PIECEWISE`, capture through 64 rows |
| Scheduling | 6,019 batched tokens, 16 sequences, OMP2 |
| Maximum request context | 262,144 tokens, including generated tokens |
| GPU prefix cache | Enabled; repeated prompts and shared SYSTEM/developer prefixes qualified |
| LMCache RAM/disk restore | Installed in the image, but **not enabled or qualified for Qwen** |
| Image/video inputs | Disabled by this text-only recipe; not qualified here |

**Do not inherit `VLLM_LM_HEAD_A16=0` from the image.** That inherited setting
produced near-zero MTP draft acceptance at C8/C16 in three repeated diagnostic
sweeps. The explicit value of one restored normal acceptance without changing
the target vocabulary head. The kernel-level cause of the W4A4 diagnostic is
not established; the W4A16 configuration is the qualified choice.

The TP1 qualification reported **859,808 usable logical KV tokens**, with a
small allocation variation between boots. This is a pool shared by requests,
not the maximum context of one request. A nominal physical-blocks-times-page
calculation reported about 13.1 million; that is **not usable capacity** for
this hybrid cache. Trust the engine's logical-capacity startup line. The
requested block size is 64, but the measured effective hybrid pages are 3,008
tokens; the recipe does not manually force a different geometry.

## Measured llmbench performance

Status: **qualified** for the exact TP1/MTP3 text configuration above. One
RTX PRO 6000 Blackwell Workstation GPU, 600 W limit, **stock clocks: memory
offset 0, graphics offset 0**. These are not +6000 results.

The [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
decode test uses ordinary reasoning, temperature 1, top-p 0.95, top-k 20 and
respects end-of-sequence (EOS). Context `0` means a short chat prompt, rendered
as 119 input tokens in this checkpoint, not literally zero input tokens.
Each concurrency has three 30-second measurements after warmup.

| Concurrent requests | Total output tok/s, median [range] | Request-level speculative steps/s | Effective accepted length |
|---:|---:|---:|---:|
| C1 | **171.79** [165.74–184.19] | 84.86 | 2.031 |
| C8 | **622.04** [614.71–631.60] | 326.22 | 1.916 |
| C16 | **933.81** [933.04–943.86] | 473.97 | 1.991 |

**C8/C16 output is the sum across all clients, not speed per chat.** TP1 still
means one GPU. Speculative steps at concurrency above one sum request-level
verification rounds, not physical GPU batch forwards. Effective accepted
length includes the verifier/bonus token.

Cold prefill uses exactly 32,768 token IDs and one output token, two warmups
and five measured requests, all with zero cached input tokens:

| 32K prefill metric | Input tok/s, median [range] |
|---|---:|
| Engine request-prefill accounting | **14,839.30** [14,814.37–14,894.25] |
| Complete one-output-token HTTP wall time | **14,690.93** [14,660.02–14,744.56] |

Neither metric is pure GPU kernel time. TP2, no-MTP, PLE-offload-disabled and
+6000 throughput are **not measured for this image**. An overclocked instance
passing a short response check is not a completed performance qualification.

The same-GPU reference/R28.1/reference comparison and individual samples are
in the [TP1 qualification report](qwen38-flash-next/validation/r28.1-tp1.md).
It also records the bounded cache/output checks and two generated-code test
defects; serving qualification is not a general model-quality certificate.

### Run the decode benchmark

On an otherwise idle server, with Python and `uv` installed:

```bash
curl -fL -o llm_decode_bench.py \
  https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/80d1f1b0ab9830c3fd8a22c42f461c40cbc7cf96/llm_decode_bench.py

for QWEN_REPEAT in 1 2 3; do
  uv run --no-project --with httpx --with rich --with psutil python llm_decode_bench.py \
    --host 127.0.0.1 --port 8000 --model Qwen3.8-Flash-Next \
    --contexts 0 --concurrency 1,8,16 --duration 30 \
    --decode-warmup-seconds 10 --max-tokens 32768 \
    --temperature 1 --respect-eos --skip-prefill \
    --display-mode plain --no-resume --output "qwen-decode-$QWEN_REPEAT.json" < /dev/null
done
```

Use a fresh output directory to preserve previous samples. Keep the same GPU,
clock offsets, model revision, launch arguments and benchmark revision for an
A/B comparison. Check errors, filled concurrency and loop flags before using
a throughput number. The source revision in the download URL freezes the
measured benchmark; the serving model remains an ordinary Hugging Face name.
