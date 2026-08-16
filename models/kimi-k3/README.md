# Kimi-K3 serving on RTX PRO 6000 Blackwell

Status: **qualified** for the serving profiles and limits identified on this
page.

This page specifies source-locked vLLM serving for the official
`moonshotai/Kimi-K3` MXFP4 checkpoint and the
`lukealonso/Kimi-K3-QSRT-K2` checkpoint. The qualified host contains 16
NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs connected through
PCIe Gen5 x16.

Related evaluation specifications:

- [Distribution-fidelity reference: 1,024 contexts × 2,048 tokens](distribution-fidelity-1024x2048.md)
- [AA-LCR reproduction](aa-lcr-reproduction.md)
- [Official MXFP4 versus QSRT K2 AA-LCR](aa-lcr-official-mxfp4-vs-qsrt-k2.md)
- [QSRT K2 TP16/DCP8 serving and checkpoint fidelity](qsrt-k2-tp16-dcp8.md)
- [RedHatAI BF16 DSpark qualification](redhat-dspark-dcp16.md)

## Qualified artifact

| Object | Durable identifier |
|---|---|
| Docker image | `voipmonitor/vllm:kimi-k3-ii-vllmf21a391-b12x3be8bc7-cu133-torch213-20260816-r2` |
| Registry digest | `sha256:9230c19c6b16ca6216613360619b0cca2356dba65c2297c99817750b3f9e4b83` |
| Docker recipe | [`blackwell-llm-docker@2c469ba`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/2c469ba2c54827d82b96b57450374b9c46f163ac) |
| Runtime base | `voipmonitor/vllm@sha256:01b973d1ae132882bcc1bf62ea232f6aabe649dd4a89b961d81f3c41cc53f971` |
| vLLM base | `dev/infernal-invocation@ad848fc4141f201489db18d5453c50b312245a0a` |
| vLLM composed tree | `f21a391de0a1b127c93ac718fd7d1818f025317b` |
| B12X base | `master@e68f812f15e6b06420cc649eb9caccfa42d1b9c4` |
| B12X composed tree | `3be8bc74d6813223b5be732a4c865401a693f5f5` |
| Official target | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| QSRT target | `lukealonso/Kimi-K3-QSRT-K2@3b98114115f1d41ce7963ba346c3fca19918b0bd` |
| DSpark draft | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |
| DFlash draft | `modal-labs/Kimi-K3-DFlash@c192d15a43407bf758b5ae0880d5c72052fef1de` |

The image uses CUDA 13.3, PyTorch 2.13.0, patched NCCL 2.31.2,
FlashInfer 0.6.18+cu133, InstantTensor 0.1.9, CUTLASS DSL 4.6.2, and
XGrammar 0.2.5. No source checkout or launcher bind mount is required.

## Qualified profiles

Measurements use one active request. The normalized decode protocol uses a
stored 256-token prompt and generates 1,024 tokens. Speculative output rate
depends on generated content, so target cycles and acceptance are reported
with decode throughput.

| Target and topology | Entrypoint | Physical target KV tokens | Decode tok/s median | Target cycles/s median | Acceptance median |
|---|---|---:|---:|---:|---:|
| Official MXFP4, TP16/DCP16, no speculation | `/usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii` | 1,460,937 | 55.28 | n/a | n/a |
| Official MXFP4, TP16/DCP16, DSpark K7 | `/usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii` | 1,057,049 | 104.21 | 29.676 | 0.357 |
| Official MXFP4, TP16/DCP16, DFlash K7 | `/usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii` | 1,048,576 | 91.10 | 28.664 | 0.311 |
| QSRT K2, TP8/DCP8, no speculation, one active sequence | `/usr/local/bin/serve-kimi-k3-qsrt-nospec` | 1,072,139 | 43.68 | n/a | n/a |

The no-speculation value is the median from an eight-run recheck of the exact
published image digest after all 16 GPUs had entered idle P8 before the first
request. The original qualification receipt recorded 52.95 tok/s and a
same-window control recorded 52.97 tok/s. The same image, input tokens, output
hash, and benchmark protocol later produced 55.275 tok/s. The difference is
execution-state dependent; its hardware or driver cause has not been isolated.
The 52.95 tok/s result remains part of the immutable functional qualification
record but is not the representative no-speculation throughput.

The no-speculation recheck is recorded in
[`validation/nospec-p8-recheck-20260816.json`](validation/nospec-p8-recheck-20260816.json).

The coding acceptance prompt `Write a Python script that implements the Sieve
of Eratosthenes.` produced median rates of 126.78 tok/s with DSpark and 149.36
tok/s with DFlash. All six measured outputs contained valid Python and no CJK
ideographs.

The consolidated machine-readable qualification receipt is
[`validation/source-locked-runtime-20260816.json`](validation/source-locked-runtime-20260816.json).

## Unpublished merge-candidate qualification

Status: **qualified as a source overlay; not present in the published Docker
image**.

[vLLM #387](https://github.com/local-inference-lab/vllm/pull/387) commit
`fe2416a7f9bc562c33fc35607d99c4febf895b0c` removes the generic empty-row mask
when B12X dense MLA and its DCP reduction provide the required neutral
empty-shard semantics. This removes 192 redundant CUDA kernel launches per
Kimi-K3 decode token across 24 dense-MLA layers.

With the published Docker runtime and source overlays, the no-speculation
profile retained 1,460,937 physical target KV tokens and measured 55.925 tok/s.
An Infernal-vLLM source control under the same runtime and B12X source measured
56.034 tok/s. The patched, unpatched, and control runs produced the same output
hash. These measurements qualify the pull-request source composition, not the
published Docker image by itself.

## Runtime invariants

The official target retains checkpoint MXFP4 routed-expert weights. The
target-only profile leaves dense and KDA projection weights in BF16. The
DSpark and DFlash profiles convert eligible target KDA projections and draft
linear weights to MXFP8 while loading; routed experts remain checkpoint
MXFP4. Target verification continues to define accepted output.

All profiles use FP8 target KV cache, B12X dense MLA decode, B12X routed MoE,
B12X DCP all-to-all, and Triton KDA prefill. Prefix caching is disabled.

The target loader uses InstantTensor AIO with a 512 MiB staging ring,
`INSTANTTENSOR_COPY=0`, and a default free-memory limit of 0.6. The loader
streams source tensors and does not retain a second complete checkpoint in
CPU memory.

Every launcher enforces:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset NCCL_GRAPH_FILE
```

The speculative profiles opt into the B12X TP16 equal-quarter all-reduce with
`B12X_PCIE_ALLREDUCE_ALGORITHM=island_rs`. The dispatcher keeps hierarchical
all-reduce for tensors up to 7,168 elements and for unaligned shapes. Larger
aligned tensors use island reduce-scatter/all-gather. The target-only profile
uses the default hierarchical policy.

FlashKDA prefill is **research-only** for the one-million-token memory
profile. Its SM120 module consumed approximately 3.74 GiB per GPU during the
measured load, compared with approximately 0.12 GiB for Triton KDA, while the
full target and physical one-million-token cache left approximately 0.45 GiB
of headroom. The qualified profiles therefore set
`--kda-prefill-backend triton`.

## Cache isolation

Each profile requires an independent writable `/cache/jit` directory because
the graph shapes and generated kernels differ.

```bash
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/nospec
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/dspark
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/dflash
mkdir -p /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp8-nospec

export KIMI_IMAGE='voipmonitor/vllm@sha256:9230c19c6b16ca6216613360619b0cca2356dba65c2297c99817750b3f9e4b83'
unset NCCL_GRAPH_FILE
```

The commands below expose the API on host port 8001 through host networking.

## Start official MXFP4 without speculation

```bash
docker run -d \
  --name kimi-k3-mxfp4-nospec \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/nospec:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-nospec-ii \
  "$KIMI_IMAGE"
```

The served model name is `Kimi-K3-MXFP4-DCP16-1M`.

## Start official MXFP4 with DSpark

This profile is the qualified default for high single-request coding
throughput with at least one million physical target KV tokens.

```bash
docker run -d \
  --name kimi-k3-mxfp4-dspark \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/dspark:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-dspark-ii \
  "$KIMI_IMAGE"
```

The served model name is `Kimi-K3-MXFP4-DSpark7-DCP16-1M`.

## Start official MXFP4 with DFlash

```bash
docker run -d \
  --name kimi-k3-mxfp4-dflash \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/dflash:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-full-mxfp4-dflash-ii \
  "$KIMI_IMAGE"
```

The served model name is `Kimi-K3-MXFP4-DFlash-DCP16-1M`.

## Start QSRT K2 on TP8/DCP8

The one-million-token TP8 profile is qualified only for one active sequence
and a 2,048-token scheduler budget.

```bash
docker run -d \
  --name kimi-k3-qsrt-k2-tp8-nospec \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --network host \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e PORT=8001 \
  -e TP_SIZE=8 \
  -e DCP_SIZE=8 \
  -e KV_CACHE_MEMORY_BYTES=1950000000 \
  -e MAX_MODEL_LEN=1048576 \
  -e MAX_NUM_SEQS=1 \
  -e MAX_NUM_BATCHED_TOKENS=2048 \
  -e SERVED_MODEL_NAME=Kimi-K3-QSRT-K2-NoSpec-TP8-DCP8-1M \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/kimi-k3-cache/source-locked-r2/qsrt-tp8-nospec:/cache/jit \
  --entrypoint /usr/local/bin/serve-kimi-k3-qsrt-nospec \
  "$KIMI_IMAGE"
```

`MAX_NUM_SEQS=8` with a 4,096-token scheduler budget and a 1.9–1.95 GB KV
allocation per GPU is **unsupported** on TP8 because CUDA Graph warmup exceeds
available GPU memory.

## Readiness and fixed-token decode

Wait for the health endpoint and verify the served identity:

```bash
until curl -fsS http://127.0.0.1:8001/health >/dev/null; do sleep 5; done
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

Measure target-only decode with the stored token sequence:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-nospec-decode.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-MXFP4-DCP16-1M \
  --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
  --prompt-tokens 256 \
  --max-tokens 1024 \
  --warmups 2 \
  --runs 8 \
  --output-dir /tmp/kimi-k3-nospec-normalized
```

Measure DSpark or DFlash with target-cycle and acceptance accounting:

```bash
python3 models/kimi-k3/tools/benchmark-kimi-k3-dspark-decode.py \
  --url http://127.0.0.1:8001 \
  --model Kimi-K3-MXFP4-DSpark7-DCP16-1M \
  --token-file models/kimi-k3/tools/decode-baseline-256-token-ids.json \
  --prompt-tokens 256 \
  --max-tokens 1024 \
  --warmups 2 \
  --runs 8 \
  --output-dir /tmp/kimi-k3-dspark-normalized
```

The token file SHA-256 is
`64c2ff11a83dda71caea12ea10b16511b0114c15973b949cde3ebd2bc4c712af`.

## Source composition

The Docker recipe composes hash-verified patches from minimal review units.
The pull-request links below identify the review units, while the immutable
source revisions embedded in the image are defined by the linked Docker recipe
and its integration lock files. Mutable pull-request heads may contain commits
that are not present in the image. In particular, the image does not contain
vLLM #387 commit `fe2416a7f9bc562c33fc35607d99c4febf895b0c`, B12X #224, or
vLLM #400.

| Repository | Pull requests | Resulting behavior |
|---|---|---|
| vLLM | [#382](https://github.com/local-inference-lab/vllm/pull/382), [#383](https://github.com/local-inference-lab/vllm/pull/383), [#384](https://github.com/local-inference-lab/vllm/pull/384), [#385](https://github.com/local-inference-lab/vllm/pull/385), [#386](https://github.com/local-inference-lab/vllm/pull/386), [#387](https://github.com/local-inference-lab/vllm/pull/387), [#388](https://github.com/local-inference-lab/vllm/pull/388), [#389](https://github.com/local-inference-lab/vllm/pull/389), [#390](https://github.com/local-inference-lab/vllm/pull/390), [#391](https://github.com/local-inference-lab/vllm/pull/391) | Bounded loading, Kimi cache geometry, DCP collectives, projection sharding, fused routed-MoE transport, dense MLA DCP, external draft runtimes, bounded DSpark state, vocabulary-sharded DSpark sampling, and TP-aware all-reduce limits |
| B12X | [#215](https://github.com/local-inference-lab/b12x/pull/215), [#220](https://github.com/local-inference-lab/b12x/pull/220) | Correct unaligned multi-row FP8 output storage and opt-in size-routed TP16 equal-quarter all-reduce |

The aggregate vLLM PR #317 and aggregate B12X PR #198 are not source inputs to
the image.

## Build from pinned source

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 2c469ba2c54827d82b96b57450374b9c46f163ac

RELEASE_DATE=20260816 \
REVISION=r2 \
./build-kimi-k3-qsrt-tp16-runtime.sh
```

The builder verifies the base-image identity, integration patch hashes,
resulting vLLM and B12X trees, imported source locations, launcher syntax, and
the FlashAttention CuTeDSL wrapper hash. The expected output tag is
`voipmonitor/vllm:kimi-k3-ii-vllmf21a391-b12x3be8bc7-cu133-torch213-20260816-r2`.

## Qualification boundaries

- The official profiles are qualified for TP16/DCP16 and one active request.
- The QSRT K2 TP8/DCP8 profile is qualified for one active request with a
  2,048-token scheduler budget.
- The physical KV capacity is not evidence of quality at a one-million-token
  context. Long-context accuracy requires a separate evaluation.
- Speculative decode rates are output-dependent. Compare runtimes by target
  cycles as well as emitted tokens per second.
- Triton KDA is qualified for the listed memory profiles. FlashKDA remains
  research-only under the one-million-token memory constraint.
