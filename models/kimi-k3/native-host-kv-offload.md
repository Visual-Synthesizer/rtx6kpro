# Kimi-K3 Native Host KV Offload

Status: **qualified** for a TP16/DCP16 Kimi-K3 runtime assembled from vLLM
pull requests [#441](https://github.com/local-inference-lab/vllm/pull/441)
and [#443](https://github.com/local-inference-lab/vllm/pull/443).

Deployment status: **research-only** until vLLM #443 is included in a
published image. The command below mounts two Python modules from an exact
source checkout over the published base image. The official MXFP4 target,
Inferact DSpark draft, and compiled GPU kernels remain those of the base image.

This profile stores reusable KV blocks in vLLM's native RAM backend. It does
not start LMCache. A completed-request replay-tail mapping restores a common
768-token boundary across the target and speculative-draft cache groups, so a
partially filled 12,288-token target page does not force recomputation of the
whole page.

The machine-readable qualification record is
[`validation/native-host-kv-offload-tp16-20260819.json`](validation/native-host-kv-offload-tp16-20260819.json).

## Source and runtime identity

| Component | Immutable identity |
|---|---|
| Base image | `voipmonitor/vllm:kimi-k3-production-dspark-lmcache-clean-vllm726b234-b12x4fd20fa-cu133-torch213-20260819-r5` |
| Docker digest | `sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa` |
| Docker recipe | `local-inference-lab/blackwell-llm-docker@0b6dd14369588b894cd0ce9fe50c783be41d3a8e` |
| vLLM composition base | `dev/infernal-invocation@337ef76dcd30198d8dd47f6c9e61ae1d8be73656` |
| Draft-group metadata | vLLM #441 commits `8164755964` and `9a8aa110a7` |
| Completed replay tails | vLLM #443 commit `2be610a9cc` |
| Combined vLLM tree | `bcc3d52dd160a86f592fa1270cb8c40a7f850b16` |
| B12X tree | `4fd20fa4bf81c476d61af9dcd11d23cb6dc1ad5a` |
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Draft checkpoint | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

The base image already contains vLLM #441. Both source modules are mounted
from one combined checkout to make the active source identity explicit:

```text
vllm/v1/core/kv_cache_utils.py
  SHA-256 ccf1f759cbb6dc86f8101fd1f3591bf2943bc3ff86f703671f4feb0acf00a91e
vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
  SHA-256 304794f90fdc720687b2ea742bdfab78987a78bb368b0e31468c621eedd1de7f
```

## Cache contract

The TP16/DCP16 Kimi-K3 profile has 16 target-model KV groups and one
speculative-draft KV group. The draft group uses 768-token chunks. Each target
group uses a 12,288-token physical page. vLLM #441 preserves the draft
annotation through cache planning and DCP grouping without a topology-specific
special case.

vLLM #443 permits a completed request to publish the immutable part of a
larger target page at the shared 768-token boundary. A replay-tail hit requires:

- one completion-only key for every partially filled larger page;
- contiguous keys for full-attention groups after the ordinary complete hit;
- the terminal window for windowed or recurrent groups;
- one later complete draft key as a stability witness.

Aborted requests do not publish replay-tail entries. Unsupported and uniform
cache layouts retain the complete-chunk behavior.

## Prepare the exact source checkout

```bash
git clone https://github.com/local-inference-lab/vllm.git /opt/kimi-native-vllm
cd /opt/kimi-native-vllm
git checkout --detach 337ef76dcd30198d8dd47f6c9e61ae1d8be73656
git cherry-pick \
  8164755964 \
  9a8aa110a7 \
  2be610a9cc

test "$(git rev-parse HEAD^{tree})" = \
  bcc3d52dd160a86f592fa1270cb8c40a7f850b16
```

The cherry-picked commit ID depends on Git committer metadata; the verified
tree ID is the source identity.

## Start the native-offload profile

The Hugging Face cache must contain the pinned target and draft snapshots.
LMCache and native KV offload must not be enabled in the same process.

```bash
docker pull voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa

mkdir -p /mnt/luke/kimi-k3-cache/native-bcc3d52-4fd20fa

docker run -d \
  --name kimi-k3-native-replay-tail-tp16 \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e LMCACHE_MODE=off \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e HOST=127.0.0.1 \
  -e PORT=8001 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /mnt/luke/kimi-k3-cache/native-bcc3d52-4fd20fa:/cache/jit:rw \
  -v /opt/kimi-native-vllm/vllm/v1/core/kv_cache_utils.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py:ro \
  -v /opt/kimi-native-vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:/opt/venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:ro \
  voipmonitor/vllm@sha256:bd8a4be5e87c89f37548ee0502c1a0dc186e9058d57f3278927c1ef5d01e65fa \
  --kv-offloading-size 32 \
  --kv-offloading-backend native
```

The runtime allocates 32 GiB of process-shared host KV memory. Its GPU cache
contains 1,033,126 token positions and the API accepts requests up to 1,000,000
tokens.

```bash
docker logs -f kimi-k3-native-replay-tail-tp16
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models | jq .
```

The startup log must contain both messages:

```text
KV offloading: EAGLE/MTP draft attention groups [16] detected.
KV offloading: completed-request replay tails use a 768-token boundary
```

## Restore qualification

The measured request contains 134,219 prompt tokens and five images. A cold
request populated the 32 GiB native host tier. Each measured repetition cleared
the GPU prefix cache while retaining the native host cache.

| Metric | Complete target pages | 768-token replay tail | Change |
|---|---:|---:|---:|
| Host-hit tokens | 122,880 | 132,864 | +9,984 |
| Recomputed tokens | 11,339 | 1,355 | -88.1% |
| Engine prefill, median | 9.239 s | 1.270 s | -86.3% |
| API TTFT, median | 9.544 s | 1.577 s | -83.5% |
| Native H2D time, median | 0.166 s | 0.173 s | +0.007 s |
| End-to-end request, median | 12.387 s | 3.957 s | -68.1% |
| Host data after cold seed | 12,706,283,520 B | 13,361,135,616 B | +624.5 MiB |

The H2D transfer is not the source of the former 9-second delay. The replay
path copies 3,875,291,136 bytes in a median 0.173 seconds; almost all saved time
comes from avoiding 9,984 prompt-token recomputations.

All three replay repetitions restored exactly 132,864 tokens, returned HTTP
200, emitted a terminal SSE event, and contained neither a stream error nor a
Kimi protocol control marker.

| Receipt | SHA-256 |
|---|---|
| `host-restore-02/receipt.json` | `620d6ac32b796148e9bbeb82a22e69a9e1c01e4077137e357ff37eb60e639d94` |
| `host-restore-03/receipt.json` | `694fd77dd9a9f3c4684fb571f4afc7515d9e683496b990d520d1e622107ee027` |
| `host-restore-04/receipt.json` | `804424b608a881bf45b475e41326c1dbe0213339c862a3a4ffd778068a7214b0` |

Host-local evidence is stored under:

```text
/mnt/luke/kimi-k3-runs/native-replay-tail-tp16-20260819
```

## Validation and limits

- Seven focused scheduler tests cover mixed cache geometry, missing completion
  and draft keys, full-attention holes, local-prefix deltas, partial-page
  stores, load mappings, and recurrent-boundary retention.
- Cache-spec promotion and DCP8/DCP16 grouping are covered by vLLM #441 tests.
- TP16/DCP16 full-model serving is qualified. TP8 grouping is unit-tested;
  TP8 full-model serving is unqualified.
- Native host storage is volatile and does not survive container termination.
- The measurement covers one active sequence and a RAM-only host tier.
- Filesystem-backed native KV storage is unsupported by this qualification.
- The source-mounted deployment remains research-only until vLLM #443 is
  compiled into a published image.
