# GLM-5.3-Flash: the release recipe as a plain `vllm serve` command

`PRESET=glm53-tp2` on `ghcr.io/local-inference-lab/vllm:release-10152026` is a
thin launcher around one `vllm serve` process. This page gives that process as
a plain command, for people who want to run or inspect the recipe without the
preset launcher. The script is
[`glm53-tp2-plain-vllm.sh`](glm53-tp2-plain-vllm.sh); run it as is, or copy the
`docker run` command out of it.

```bash
curl -fsSLO https://raw.githubusercontent.com/local-inference-lab/rtx6kpro/master/models/glm-5.3-flash/glm53-tp2-plain-vllm.sh
GPUS=0,1 PORT=8000 HF_CACHE=model-cache bash glm53-tp2-plain-vllm.sh
docker logs -f glm53-plain          # ready when "Application startup complete" appears
curl -s http://127.0.0.1:8000/v1/models
```

What the command contains:

- `vllm serve local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD --revision 96f27f8c…`
  (branch `qad2-kda-w8a16-csf`): vLLM downloads the pinned revision into the
  mounted Hugging Face cache itself (about 60 GB, public repository, no token).
- `--quantization modelopt_mixed --load-format instanttensor`: the checkpoint's
  `config.json` carries ModelOpt recipes that mark the FP4-CSF expert scales
  (`csf_32x2_v1`); FlashInfer's B12X loader in this image expands them on the
  GPU.
- TP2, DCP2, MTP3 with B12X drafter experts, eight request slots, a 4,096-token
  prefill budget, 7,650,410,496 bytes of FP8 KV cache per GPU (1,844,162
  tokens), `temperature 1.0` and `top_p 0.95` as server defaults,
  `reasoning_effort max` through the chat template, the `glm45` reasoning
  parser and `glm47` tool parser.
- 61 environment variables the launcher sets from its profile, preset and
  hardware layers, among them `VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill`
  (BF16 activations for the MXFP8 KDA projections in prefill),
  `VLLM_GLM53_MTP_DRAFT_HEAD=bf16`, `VLLM_B12X_MOE_FP4_FORCE_A16=1` with
  `B12X_W4A16_A4_PREFILL=1` (routed experts W4A16 in decode, NVFP4 activations
  in prefill), the B12X PCIe all-reduce and the NCCL settings for PCIe hosts.
- The container still runs the image's CUDA/NCCL bootstrap (`vllm serve …` as
  the container command keeps it); only the model-profile logic is bypassed.

What differs from `PRESET=glm53-tp2`:

- The preset's default adds an LMCache RAM tier (a second process, 64 GiB of
  host RAM) that restores conversations evicted from the GPU cache. The plain
  command uses the GPU prefix cache only, which is the preset's
  `CACHE_MODE=vram` form. The KV size is the same.
- No automatic restart (`--restart on-failure` is left out so that a failing
  configuration stops with its error in `docker logs`).
- JIT cache directories use the libraries' defaults under `/cache`
  (`XDG_CACHE_HOME`), not the launcher's runtime-keyed paths.

Why `--print-config` output cannot be pasted directly: it shows the launcher's
own option names `--load-format nvfp4_csf --quantization nvfp4_csf` and the
repository id. At execution the launcher resolves the checkpoint snapshot and
rewrites those two options to `instanttensor` and `modelopt_mixed`; vLLM has
no quantization method named `nvfp4_csf`. The script applies the same rewrites.

## Verification

Measured 2026-10-16 on one 16-GPU RTX PRO 6000 Blackwell host, both servers on
the same image digest (`sha256:745feb91…`, `karmic-kraken-20261015.8` =
`release-10152026`): `PRESET=glm53-tp2 CACHE_MODE=vram` on two GPUs against
this script on two other GPUs.

- The API server's argv (read from `/proc`) is identical apart from the model
  argument (the launcher passes the resolved snapshot directory and drops
  `--revision`; the script passes the repository id with `--revision`) and the
  port.
- The process environment is identical for every `VLLM_*`, `B12X_*`, `NCCL_*`,
  `CUDA_*`, `TORCH*`, `OMP_*`, `PYTORCH_*` and `GLOO_*` variable; only the JIT
  cache paths differ.
- vLLM's cache configuration is identical: 1,844,162 KV tokens, 554 blocks of
  2,048 tokens, Mamba block 256, FP8 KV, `recurrent_checkpoint_policy
  request_boundaries`.
- A greedy request (`temperature 0`, 102 output tokens) returned byte-identical
  text from both servers; MTP drafting was active on both.
- Weights loaded in 39 s from an already populated Hugging Face cache.

Changing `MODEL` to a mirror of this checkpoint requires `REVISION` set to the
mirror's own commit (or a mirror that copies revision `96f27f8c` file for file).
