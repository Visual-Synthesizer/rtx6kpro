# Kimi-K3 Native Host KV Offload

Status: **qualified** for TP16/DCP16 on 16 RTX PRO 6000 Blackwell GPUs.

This profile serves the official Kimi-K3 MXFP4 target with the Inferact
DSpark draft and stores reusable KV blocks in native vLLM host memory. It uses
the same one-million-token GPU KV geometry as the production LMCache profile,
but does not start LMCache.

The machine-readable qualification record is
[`validation/native-host-kv-offload-tp16-20260819.json`](validation/native-host-kv-offload-tp16-20260819.json).

## Published artifact

| Component | Immutable identity |
|---|---|
| vLLM image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-clean-vllm95114d4-b12x4fd20fa-cu133-torch213-20260819-r4` |
| Docker digest | `sha256:67936d155e318122a545beae97579146b086a62842bdb04b52070a4b18b45c9a` |
| Docker recipe | `local-inference-lab/blackwell-llm-docker@71d02d23a64228d73032effe27e5abeccded5258` |
| vLLM merge commit | `1057b63a6844b7e3816e4eaeac06d913a09c3e9d` |
| vLLM tree | `95114d4e367b24a923e77af33b38f3c795c3f000` |
| B12X tree | `4fd20fa4bf81c476d61af9dcd11d23cb6dc1ad5a` |
| vLLM fix | [`local-inference-lab/vllm#441`](https://github.com/local-inference-lab/vllm/pull/441) |

The image contains compiled installed packages and no source overlay. Its OCI
labels record every vLLM and B12X pull-request head used by the build.

## Cache-group contract

Kimi-K3 has 16 target-model KV groups and one speculative-draft KV group in
the TP16/DCP16 profile. Only the draft group has non-causal multi-token decode
semantics and requires a volatile trailing cache region.

The cache planner preserves
`AttentionSpec.non_causal_multi_token_decode` when it promotes an MLA cache
specification for allocation. It then marks a cache group as a draft group
only when one of that group's layer specifications has the non-causal
multi-token property. The rule reads model metadata and does not contain a
TP8, TP12, TP16, DCP8, or DCP16 special case.

The scheduler reports the resulting TP16 classification as:

```text
KV offloading: EAGLE/MTP draft attention groups [16] detected.
```

Unit tests cover DCP8 and DCP16 grouping. Full-model execution is qualified on
TP16/DCP16; TP8 full-model execution is unqualified.

## Start the native-offload profile

The Hugging Face cache must contain the pinned Kimi-K3 and Inferact DSpark
snapshots. LMCache and native KV offload must not be enabled in the same
process.

```bash
docker pull voipmonitor/vllm@sha256:67936d155e318122a545beae97579146b086a62842bdb04b52070a4b18b45c9a

mkdir -p /mnt/luke/kimi-k3-cache/kimi-k3-production-clean-95114d4-4fd20fa

docker run -d \
  --name kimi-k3-native-offload-tp16 \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  -e LMCACHE_MODE=off \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/kimi-k3-production-clean-95114d4-4fd20fa:/cache/jit:rw \
  voipmonitor/vllm@sha256:67936d155e318122a545beae97579146b086a62842bdb04b52070a4b18b45c9a \
  --kv-offloading-size 32 \
  --kv-offloading-backend native
```

The runtime allocates 32 GiB of process-shared host KV memory. The GPU cache
contains 1,033,126 token positions and the API accepts requests up to
1,000,000 tokens.

Check readiness with:

```bash
docker logs -f kimi-k3-native-offload-tp16
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

## Restore qualification

The measured request contains 134,219 prompt tokens and five images. A cold
request populates native host memory. Each measured repetition clears the GPU
prefix cache while retaining the native host cache.

| Metric | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Host-hit tokens | 122,880 | 122,880 | 122,880 | 122,880 |
| Recomputed tokens | 11,339 | 11,339 | 11,339 | 11,339 |
| Engine prefill time | 9.250 s | 9.271 s | 9.196 s | 9.250 s |
| Engine TTFT | 9.611 s | 9.587 s | 9.499 s | 9.587 s |
| Native H2D time | 0.095 s | 0.168 s | 0.161 s | 0.161 s |
| End-to-end request time | 12.202 s | 12.857 s | 12.843 s | 12.843 s |

Every restore transferred 3,705,421,824 bytes from host memory. All responses
returned HTTP 200, emitted a terminal SSE event, and contained neither a
stream error nor a Kimi protocol control marker.

The same request before cache-group correction restored only 73,728 tokens,
recomputed 60,491 tokens, and had a 44.593-second median engine TTFT. The
corrected planner restores 49,152 additional tokens and reduces median engine
TTFT by 78.5%. The corrected native profile also measured lower TTFT than the
20.151-second LMCache control for this request. These numbers apply to the
specified single-request, in-memory restore workload; they do not establish a
general advantage for either cache implementation.

Host-local evidence is stored under:

```text
/mnt/luke/kimi-k3-runs/native-offload-ab-20260819
/mnt/luke/kimi-k3-runs/native-offload-source-locked-r4-tp16-20260819
```

## Rebuild the image

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 71d02d23a64228d73032effe27e5abeccded5258

./build-kimi-k3-production-clean-runtime.sh
```

The source lock builds vLLM tree
`95114d4e367b24a923e77af33b38f3c795c3f000`, B12X tree
`4fd20fa4bf81c476d61af9dcd11d23cb6dc1ad5a`, and LMCache tree
`e045d729bc5c4c63a40e13d032f42923de97812f`. LMCache remains installed so the
same image can run the documented LMCache profile, but `LMCACHE_MODE=off`
keeps it inactive in the native-offload profile.

## Operational limits

- TP16/DCP16 full-model serving is qualified.
- DCP8 cache grouping is unit-tested; TP8 full-model serving is unqualified.
- Native host storage is volatile and does not survive container termination.
- The qualification covers one active sequence and a RAM-only host tier.
- Filesystem-backed native KV storage is unsupported by this qualification.
- Native KV offload and LMCache are mutually exclusive for a serving process.

