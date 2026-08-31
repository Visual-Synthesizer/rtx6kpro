# ML Primitive Glossary

This page defines common LLM serving and kernel terms using a small set of
shape symbols. It is written for reading performance notes, kernel logs, and
optimization discussions around attention, KV cache, MLA, MoE, and
quantization.

## Shape Symbols

| Symbol | Meaning |
|---|---|
| `T` | Active tokens |
| `B` | Active sequences |
| `L` | Live sequence length |
| `H` | Hidden size |
| `V` | Vocabulary size |
| `E` | Experts |
| `K` | Top-k experts |
| `I` | MLP or expert intermediate size |
| `A` | Attention heads |
| `G` | KV heads |
| `D` | Head dimension |
| `D_rope` | RoPE or positional feature width |
| `D_nope` | noPE or content feature width |
| `C` | Compressed or latent KV width |

## Quick Mental Map

Attention is history work: in decode, each new token reads from the live cache,
so cost grows with `L`.

GEMM and MoE are token transformation work: their cost mostly depends on the
current active tokens and hidden dimensions, not the total conversation length.

Quantization mostly buys bandwidth: FP8 and FP4 reduce storage and memory
movement, while scale metadata preserves useful numeric range.

Parallelism chooses where to split work: TP shards tensor math, while EP shards
experts and routes tokens to expert owners.

## Glossary

| Term | Typical Shape | Meaning and Scaling Notes |
|---|---:|---|
| `Token` | Scalar position | One model position. In prefill, `T` can be thousands. In decode, `T ~= B`, usually one new token per live sequence. |
| `Activation` / hidden state | `[T, H]`, for example `[T, 4096]` | The per-token vector flowing through layers. Usually BF16/FP16; some kernels quantize internally to FP8 or FP4. |
| `Logits` | `[T, V]` | Complete raw score vector over the vocabulary. Softmax turns this into the next-token probability distribution. |
| `Softmax` | `[T, V]`, `[T, E]`, or attention scores | Converts raw scores into normalized probabilities or weights. Used for vocab sampling, routing, and attention. |
| `Linear` / `GEMM` | `[T, H] @ [H, O] -> [T, O]` | Core matrix multiply. Cost scales with `T * H * O`. |
| `Attention` | Q `[T, A, D]`; K/V cache `[B, L, G, D]` | Lets current tokens read previous tokens. Decode attention grows with live sequence length. |
| `Q / K / V` | Q `[T, A, D]`; K/V `[T or L, G, D]` | Query asks what to look for; key identifies stored positions; value is the data mixed into the output. |
| `RoPE` | Often `[T, heads, D_rope]` | Rotary Positional Embedding. It rotates Q/K features based on token position so attention scores include position information. In MLA-style caches, the RoPE part is the positional component. |
| `noPE` | Often `[T or L, heads, D_nope]` | Features without positional embedding. These Q/K dimensions are not RoPE-rotated, so they carry content-like matching information. In b12x compressed MLA, noPE cache data is commonly stored as FP8 plus scales. |
| `GQA` | Q `[T, A, D]`; K/V `[B, L, G, D]`, with `G < A` | Grouped-Query Attention. Many query heads share fewer KV heads, preserving query head count while reducing KV cache size and decode read bandwidth. |
| `KV cache` | Logical `[B, L, G, D]`; paged `[blocks, block_size, G, D]` | Stores past K/V so decode avoids recomputing old tokens. Long-context decode still has to read cached history and perform score/mix work. |
| `Live sequence length` | `L` per sequence | Number of cached tokens still visible to a sequence. This is the main driver of decode attention cost. |
| `Prefill` | `T = total prompt tokens` | Prompt ingestion. Large dense batches, high parallelism, throughput-oriented kernels. |
| `Decode` | `T ~= B` | Autoregressive generation. Small token batches, long KV reads, irregular MoE routing, and latency-sensitive kernels. |
| `Speculative decoding` | `T = B * num_spec_tokens` | A draft proposes `num_spec_tokens` candidate tokens per sequence, and the target model verifies them in one forward pass. So the active token count is `B * num_spec_tokens`, not `B`. Decode then behaves more like a small prefill: larger `T` and more GEMM/MoE work per step, while KV reads still scale with `L`. |
| `Paged attention` | KV `[pages, page_size, G, D]` | Serving-friendly KV layout for variable-length sequences. KV may be BF16/FP16 or FP8. |
| `Sparse attention` / `NSA` | Selected blocks from `[B, L, G, D]` | Reads selected history blocks instead of all of `L`. The benefit grows with context length. |
| `MLA` | Shared latent/noPE cache `[B, L, C]` plus RoPE cache `[B, L, D_rope]` | Multi-head Latent Attention replaces per-head cached K/V vectors with one compressed cache vector shared across attention heads. Each head uses learned projection weights to recover its effective key/value behavior. |
| `Indexer` / `Top-k` | Logits over blocks or tokens, for example `[B, candidates]` | Scores cache blocks and selects which ones sparse attention should read. |
| `MoE` | Activations `[T, H]`; router logits `[T, E]` | Routes each token to `K` expert MLPs. Cost scales with `T * K * expert_size`, not live sequence length. |
| `Router` | `[T, E] -> ids/weights [T, K]` | Scores experts and selects the top-k experts per token. |
| `Expert` | `[tokens_for_expert, H] @ [H, I]`, then `[I, H]` | One MLP inside MoE. Prefill batches well; decode often creates tiny uneven expert batches. |
| `Residual` / norm helper | `[T, H]` | Adds or normalizes activation streams. Usually BF16. Fusing these helpers reduces memory traffic. |
| `Quantization` | Payload plus scales | Stores values in FP8/FP4 with scale metadata. Saves bandwidth and storage; accuracy depends on format and scale granularity. |
| `FP8` | Often E4M3 payloads | Used for MXFP8 GEMMs, KV cache, indexer data, and compressed MLA noPE vectors. |
| `FP4` / `NVFP4` | 4-bit payloads plus scales | Used for low-bandwidth GEMM and MoE paths. |
| `W4A16` | 4-bit weights, BF16 activations `[T, H]` | FP4/NVFP4 weights with BF16 activations and inline weight dequantization. |
| `TP sharding` | Split tensor dimensions across GPUs | Tensor parallelism shards big matrices or attention heads. It reduces per-GPU math and memory, but introduces collectives such as all-reduce or all-gather. |
| `EP sharding` | Split experts across GPUs | Expert parallelism assigns experts to GPUs. Tokens route to expert owners, usually via all-to-all communication. |

## Mental Models

### Attention Is History Work

Attention is the part of the model that looks backward. In decode, each new
token may need to compare against the sequence's existing cache, so cost grows
with `L`, the live sequence length. The KV cache avoids recomputing old keys and
values, but it does not make history free. It turns recomputation into cache
reads plus score/mix work.

### GQA Shrinks The KV Side

Standard multi-head attention has one K/V head per query head, so `G = A`. GQA
keeps `A` query heads but stores only `G` KV heads, with each KV head shared by
a group of query heads. Multi-Query Attention is the extreme case where `G = 1`.

This mostly helps decode, because the KV cache and cache reads scale with
`G * D` rather than `A * D`.

### MLA Stores One Shared Vector Instead Of Per-Head K/V

Use `A` for attention heads and `H` for hidden size. In traditional multi-head
attention, a model with `A = 16` attention heads stores separate K/V vectors for
those heads, roughly K `[B, L, A, D]` and V `[B, L, A, D]`. GQA reduces this by
storing only `G` KV heads, but it still stores expanded K/V vectors.

MLA changes what gets cached. As a mental model, it replaces many per-head K/V
vectors with a single compressed cache vector shared across all attention heads,
plus a smaller RoPE positional component. Each attention head has learned fixed
projection weights, similar to a per-head decompression matrix, that map from
the shared compressed space into that head's effective key/value space.

That is how MLA satisfies the same contract as plain attention. Plain attention
needs something it can score against the query, like `score_t = q dot k_t`, and
something it can mix after softmax, like
`sum_t softmax(score)_t * v_t`. MLA stores less data, but the head-specific
learned projections let the kernel recover the effective K/V behavior needed
for those two operations.

Optimized MLA kernels usually do not materialize decompressed per-head K/V
vectors in memory. They can absorb the decompression projection into the
attention math and operate directly on the shared compressed cache vector plus
the per-head projection weights.

For a concrete dimensionality example, a conventional GQA-style cache with
`G = 8` KV heads and `D = 128` head dimension stores both K and V, or
`2 * G * D = 2 * 8 * 128 = 2048` scalar cache values per token. A compressed
MLA-style cache might store roughly `D_nope = 448` content/noPE values plus
`D_rope = 64` positional values, or about `512` scalar cache values per token
plus scale metadata. That is about a 4x reduction in cache element count in this
example.

### RoPE And noPE Split Position From Content

RoPE is how many transformer attention layers inject token position into Q/K
matching. Positions become rotations in feature space, so the dot product
depends on where tokens are.

noPE is the part that skips that rotation. MLA-style designs can cache and
process these parts differently: the noPE portion is usually larger and more
compressible, while the RoPE portion preserves the positional signal needed for
attention scoring.

### GEMM And MoE Are Token Transformation Work

Dense linear layers and MoE layers transform the current token activations.
Their cost mostly grows with `T`, `H`, `I`, and, for MoE, `K`. They do not
directly care how long the conversation is. A decode token with 100K cached
history still has roughly the same MLP/MoE work as a decode token with 1K cached
history.

### Prefill Is Throughput, Decode Is Latency

Prefill has many prompt tokens, so kernels can run large dense tiles and keep
the GPU busy. Decode usually has one new token per sequence, so batches are
smaller, cache reads are longer, and routing is more irregular. That is why the
same model operation often needs different prefill and decode kernels.

### Speculative Decoding Inflates Active Tokens

Plain decode runs one new token per sequence, so `T ~= B`. Speculative decoding
breaks that assumption. A small draft model proposes `num_spec_tokens` candidate
tokens per sequence, and the target model verifies a whole block of them in a
single forward pass.

So the active token count under speculative decoding is
`T = B * num_spec_tokens`, not `B`. With batch size `B` and `num_spec_tokens`
draft tokens per sequence, the target step processes `B * num_spec_tokens`
positions at once. (Some frameworks verify one extra bonus token per sequence,
making it `B * (num_spec_tokens + 1)`.)

This is why a speculative decode step looks more like a small prefill than a
plain decode step: `T` is larger, so GEMM and MoE work per step grow, while KV
reads still scale with `L`. The payoff is fewer target forward passes whenever
the draft guesses well.

### Sparse Attention Trades Selection For Fewer Reads

Sparse attention adds an indexer or top-k step to decide which history blocks
matter. That extra step only makes sense if it avoids enough KV reads afterward.
The longer `L` gets, the more attractive this trade becomes.

### MoE Is Sparse In Parameters, Not Time History

MoE models may have many experts, but each token only activates a few. That
increases parameter capacity without running every expert for every token. The
hard serving problem is routing tokens efficiently, especially in decode, when
expert batches are small and uneven.

### TP Splits Math, EP Splits Experts

TP shards tensor dimensions: columns, rows, or attention heads. It reduces
per-GPU matrix size but introduces collectives.

EP shards the expert set: different GPUs own different experts. It scales MoE
capacity, but introduces token routing and load-balancing problems.

### Quantization Mostly Buys Bandwidth

FP8 and FP4 reduce memory movement and storage. Scales recover useful numeric
range, and kernels often dequantize inline. Weight quantization helps when
weights dominate bandwidth; activation and KV quantization help when runtime
tensors or long-context cache reads dominate.
