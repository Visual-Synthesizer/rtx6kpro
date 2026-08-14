# RedHatAI BF16 DSpark for Kimi-K3 with DCP16

Status: **qualified** for single-request decode with a 4,096-token server
limit. Context lengths of 128K tokens and above are **unsupported** by the
qualified memory profile.

## Purpose

This page specifies how to serve the official Kimi-K3 MXFP4 checkpoint with
the `RedHatAI/Kimi-K3-speculator.dspark` BF16 draft on 16 NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPUs. The target uses TP16/DCP16. The draft is
replicated and retains the 2,048-token sliding-attention window encoded in its
checkpoint.

## Artifact identity

| Object | Durable identifier |
|---|---|
| Target checkpoint | `moonshotai/Kimi-K3@2496450e92e425c886db095102a52a6682ca3970` |
| Draft checkpoint | `RedHatAI/Kimi-K3-speculator.dspark@46264ceaf6e011cd203f5735af5081c91ac6a235` |
| Runtime image | `voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1` |
| Runtime image digest | `sha256:974edc237f27a4eaa83a53ce4927dd176a5ad8ce4fbb8d3d689fce82348531a5` |
| vLLM compatibility fix | [`local-inference-lab/vllm#310`](https://github.com/local-inference-lab/vllm/pull/310), commit `a23c54ca2ff76ae2487e93ec92455cae7d5eae63` |
| B12X source in the image | tree `2e6092a74d2449b8f8fa65d0c980533002db76cb` |
| Weight loader | InstantTensor 0.1.9 for the target; safetensors for the draft |

The target checkpoint retains MXFP4 routed-expert weights. The server converts
the target MLA `q_proj`, `k_proj`, `v_proj`, `b_proj`, and `f_a_proj` tensors to
MXFP8 during loading. The draft weights and draft KV cache remain BF16. The
target KV cache uses FP8.

## Required vLLM behavior

The RedHatAI checkpoint uses the Qwen3 NeoX rotary layout and declares every
draft layer as `sliding_attention` with a 2,048-token window. The draft KV
cache is replicated across TP ranks even when the target KV cache is sharded
by DCP.

vLLM PR #310 enforces three invariants:

- DSpark derives its draft model and load configuration without replacing
  checkpoint rotary-layout fields.
- DFlash-family helpers copy target rotary-layout metadata when the target
  supplies that metadata; DSpark retains checkpoint metadata.
- A replicated draft KV group executes with DCP world size 1 and rank 0 and
  does not produce cross-rank decode LSE output.

Without the replicated-KV DCP normalization, a DCP16 draft reads only a
fraction of its local attention context. The failure is numerically silent and
reduces greedy block acceptance from approximately 20.7% to 0.84% in the
fixed-input diagnostic.

## Launch

Check out the vLLM fix and pull the immutable runtime image:

```bash
git clone https://github.com/local-inference-lab/vllm.git \
  /mnt/luke/vllm-k3-redhat-dspark
git -C /mnt/luke/vllm-k3-redhat-dspark checkout --detach \
  a23c54ca2ff76ae2487e93ec92455cae7d5eae63

docker pull \
  voipmonitor/vllm@sha256:974edc237f27a4eaa83a53ce4927dd176a5ad8ce4fbb8d3d689fce82348531a5

SOURCE_DIR=/mnt/luke/vllm-k3-redhat-dspark \
  models/kimi-k3/tools/launch-kimi-k3-redhat-dspark-dcp16.sh
```

The launcher binds the five Python implementation files changed by PR #310
over the immutable image. It does not modify the image or either checkpoint.
The default served model name is
`Kimi-K3-MXFP4-RedHat-DSpark7-DCP16`, and the API listens on host port 8001.

The qualified profile uses:

| Parameter | Value |
|---|---:|
| Tensor parallel size | 16 |
| Decode context parallel size | 16 |
| Draft lookahead | 7 tokens |
| CUDA graph capture shape | 8 tokens |
| Target KV allocation | 384 MiB per GPU |
| Reported target KV capacity | 6,011 tokens |
| Maximum model length | 4,096 tokens |
| Maximum batched tokens | 4,096 |
| Maximum sequences | 1 |

The draft lookahead is limited to seven because a lookahead of eight requires
a nine-token CUDA graph. Nine-token graph capture exhausted available GPU
memory with 384 MiB and 352 MiB target KV allocations. A 256 MiB target KV
allocation did not provide enough capacity for a 3,072-token request.

## Readiness and request test

```bash
docker logs -f kimi-k3-redhat-dspark-dcp16

curl -fsS http://127.0.0.1:8001/v1/models | jq .
curl -fsS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Kimi-K3-MXFP4-RedHat-DSpark7-DCP16",
    "messages": [
      {"role": "user", "content": "Write a Python script that implements the Sieve of Eratosthenes."}
    ],
    "max_tokens": 2000,
    "stream": true
  }'
```

The qualification requests omitted `temperature` and `seed`. Reasoning and
final-content streams were both included in the output-integrity checks.

## Numerical qualification

The diagnostic captured the BF16 draft input, every draft-layer output, final
hidden state, draft projection, and proposed token sequence. A CPU reference
replayed the pinned RedHatAI checkpoint using its checkpoint configuration.

| Measurement | DCP16 result |
|---|---:|
| Final-hidden cosine similarity | 0.9999585152 |
| Final-hidden normalized RMSE | 0.0096502025 |
| Projection cosine similarity | 0.9999918342 |
| Proposed tokens matching the reference | 7 of 8 |
| Reference logit margin at the differing eighth token | 0.0 |

The eighth-token difference is an exact top-logit tie in the BF16 reference.
The receipt is
`/mnt/luke/kimi-k3-runs/redhat-dspark-rope-layout-qualification-20260814/reference-comparison-dcp16-replicated-attention-fix.json`
with SHA-256
`c6af08c977f251519e35b1c1e7553c6d7f4ef49b3a08b6fbd6f0cf1506eaf570`.

## Decode qualification

Measurements used one active request on 16 RTX PRO 6000 GPUs. The normalized
protocol used a 256-token prompt, 128 generated tokens, greedy sampling, and
three measured runs. The coding protocol used the Sieve prompt shown above,
up to 2,000 generated tokens, one discarded warm-up, and three measured runs.

| Profile | Protocol | Median generation rate | Median acceptance | Emitted tokens per target cycle | Target cycles/s |
|---|---|---:|---:|---:|---:|
| Full MXFP4 target with selected MLA projections in MXFP8, no speculation | normalized | 59.20 tok/s | n/a | 1.0 | 59.20 |
| Full MXFP4 target with RedHatAI BF16 DSpark | normalized | 78.60 tok/s | 0.229 | 2.600 | 30.93 |
| Full MXFP4 target with selected MLA projections in MXFP8, no speculation | Sieve coding | 59.11 tok/s | n/a | 1.0 | 59.11 |
| Full MXFP4 target with RedHatAI BF16 DSpark | Sieve coding | 145.21 tok/s | 0.534 | 4.741 | 30.68 |

The Sieve coding gain is 2.457 times the matched no-spec target rate. All three
RedHatAI DSpark outputs contained syntactically valid Python, produced no CJK
ideographs, and passed
`sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]`.

The normalized speculative receipt is
`wikitext-redhat-dspark-k7-dcp16-graph-k8-256x128/summary.json` with SHA-256
`ee1252fbf5faba938e2064c1d2e3f67127c99e56d30e12e5acfa5221656221cf`.
The coding receipt is
`sieve-redhat-dspark-k7-dcp16-graph-k8-split-streams/summary.json` with SHA-256
`ce6b5a3d45fe7f0e9a165ca645d8263278c6044933ec362578303c952b99e9d1`.
Host-local receipt paths are relative to:

```text
/mnt/luke/kimi-k3-runs/redhat-dspark-rope-layout-qualification-20260814
```

## Limitations

- The 384 MiB target KV allocation reports 6,011 target tokens. It is not an
  AA-LCR or 1M-context profile.
- The Sieve generation rate depends on generated content and speculative
  acceptance. The normalized rate is the output-controlled comparison.
- The no-spec control and speculative target both use online MXFP8 conversion
  for five target MLA projections. These results do not measure a target whose
  dense and KDA projections all remain BF16.
- Seven-token speculation is qualified. Eight-token speculation works in
  eager execution but its nine-token CUDA graph is unsupported by the measured
  memory envelope.
