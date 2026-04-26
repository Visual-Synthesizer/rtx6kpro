# Debugging methodology — finding μs losses in `triton_mla` under vLLM + MTP

Notes for another agent (AI or human) who needs to reproduce this style of
kernel-perf diagnosis on a vLLM MLA-stack setup.

Target reader: someone who sees a Triton/MLA decode path that's slower than
expected, needs to attribute the gap to a specific layer (kernel config,
CUDA graph overhead, Python-side metadata build, CLI/deployment choice, …),
and wants to fix it.

Everything below is **what actually worked in the Kimi-K2.6 MTP session**.
See `upstream-pr-plan-review.md` for what was verified against upstream
state.

---

## 0. Mental model (read first)

A vLLM decode step on MLA has roughly these cost layers. Knowing the order
tells you which tool to reach for:

```
┌────────────────────────────────────────────────────────────────┐
│ Python scheduler step    (ms, single-threaded, CPU-bound)      │
│  ├─ metadata build                                              │
│  │   ├─ block_table prep  (← _build_decode)                    │
│  │   └─ seq_lens prep                                           │
│  └─ CG replay dispatch                                          │
├────────────────────────────────────────────────────────────────┤
│ CUDA graph replay        (μs-ms, GPU-bound)                    │
│  ├─ embedding / LN                                              │
│  ├─ 61 × attention layer                                        │
│  │   ├─ stage1 kernel       (←  _fwd_grouped_kernel_stage1)     │
│  │   └─ stage2 softmax merge (← _fwd_kernel_stage2)             │
│  ├─ 61 × MoE / MLP                                              │
│  └─ sampling / draft model                                      │
└────────────────────────────────────────────────────────────────┘
```

**Classes of perf loss you'll encounter, ranked by how often they came up
in this project**:

1. **CG not captured at all** → attention runs eager in PIECEWISE mode.
   Symptom: log line `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with
   <Backend>… setting cudagraph_mode=PIECEWISE`. Fix: change
   `_cudagraph_support` on the metadata builder (§1).
2. **Wrong Triton constexpr per CG bucket** → kernel launches too many or
   too few CTAs for the runtime workload. Symptom: one workload shape wins,
   another regresses. Fix: make the value vary by bucket (§2).
3. **Kernel tile sizes mis-tuned** → compute/shmem tradeoffs not ideal.
   Symptom: nothing obviously wrong, just "slower than it should be". Fix:
   microbench sweep (§3).
4. **Out-of-kernel overhead** → Python prep work, memcpy into CG buffers,
   CG pool size pressure. Symptom: kernel microbench shows no issue but
   e2e does. Fix: isolate with A/B (§4).
5. **Deployment CLI misconfigured** → max-model-len, max-num-batched-tokens,
   etc. pick a capture shape that hurts the real workload. Symptom: same
   codebase runs X tok/s in one config, Y tok/s in another. Fix: CLI
   isolation (§5).

---

## 1. Confirming the CG-level issue (log reading + analytical)

Start every investigation by **reading the server startup log**. Grep for:

```
grep -E "cudagraph_mode|CUDAGraphMode|CUDA graph pool" <log>
```

Key signals:

- `setting cudagraph_mode=PIECEWISE` → a backend declined FULL CG. Attention
  will run eager.
- `CUDA graph pool memory: X GiB (actual), Y GiB (estimated)` — from
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`. Actual vs estimated gap >2×
  means something's under-provisioned.

To check CG support of a backend without running it, read the builder
class declarations:

```bash
grep -n "_cudagraph_support\|query_len_support" \
  /opt/vllm/vllm/v1/attention/backends/mla/*.py
```

Compare the possible values:

| value | what it means |
|---|---|
| `AttentionCGSupport.NEVER` | no CG capture; every decode step runs eager |
| `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` | CG for plain B=N decode only, no spec-verify |
| `AttentionCGSupport.UNIFORM_BATCH` | CG for any uniform query_len (spec-verify safe) |

If you see `NEVER` on a backend that's being used in the hot path, that is
almost always where the μs are bleeding. This is the Phase 1 finding in the
Kimi session.

**Analytical estimate of the gap** (no measurement needed yet):

```
layers (target) × attention_kernel_time_per_layer × steps_per_second
```

For a 61-layer model at num_tokens=4 and ~85 ms/step of attention time: the
gap to where FlashInfer's FULL-CG path operates (~19 ms for the same
workload) is ~65 ms × 11 steps/sec = 715 ms/s = ~0.7 s lost per wallclock
second. That's the order of magnitude that turns 107 tok/s into 25 tok/s.
Matches the observed 4.3× win when the CG gap is closed.

---

## 2. Finding mis-tuned Triton `tl.constexpr` under CG (analytical CTA count)

Once CG is captured, the next class of loss is **constant values baked into
the Triton kernel that are wrong for the workload shape**.

In the Kimi case this was `CG_NUM_KV_SPLITS = 64`. For a split-KV MLA decode
kernel the stage-1 grid is

```
grid = (B, ceil(heads / BLOCK_H), NUM_KV_SPLITS)
```

To decide if a baked-in `NUM_KV_SPLITS` is right:

1. Enumerate the `(B, heads)` pairs the CG captures at.
2. For each, compute stage-1 grid block count.
3. Compare to SM count (144 on Blackwell).
4. Ratio > 2× → over-subscribed; ratio < 0.5× → under-subscribed.

Example: B=128, q_heads=128, BLOCK_H=8, NUM_KV_SPLITS=64:

```
grid = 128 × cdiv(128, 8) × 64 = 128 × 16 × 64 = 131 072 blocks
131 072 / 144 = 910× oversubscription  ← very bad
```

Fix: make NUM_KV_SPLITS a Python int that depends on B, computed once per
CG bucket. Each bucket gets its own compiled kernel (different constexpr),
but within a bucket the value is constant so CG capture stays valid.

Sanity-check the corresponding buffer shape — here `attn_logits` of shape
`(B, heads, NUM_KV_SPLITS, lora+1)`. If NUM_KV_SPLITS varies per bucket but
you want shared storage:

- Allocate at `MAX_NUM_KV_SPLITS` once.
- Pass `buf[:B, :, :num_kv_splits, :]` to the kernel each call.
- Slicing preserves parent strides → kernel pointer arithmetic is
  correct across different `num_kv_splits` values within the same
  underlying storage.

Verify slice correctness by reading the kernel: every `tl.load` should be
masked by `offs_n < split_kv_end`, and stride-based access should use
`att_out.stride(i)` consistently — never hard-coded dim sizes.

---

## 2.5. CUDA-event instrumentation — *this is how the GPU→CPU sync was found*

**This section is about targeted live-server instrumentation, not microbench.**
It's separate from §3 because the microbench only sees kernel compute time,
never metadata build time / Python scheduler overhead / CPU-GPU sync stalls
/ async-scheduling interactions. For those you need to measure **inside the
running server**, and do it with `torch.cuda.Event` instead of `.cpu()` /
`time.perf_counter()` alone to avoid perturbing what you're measuring.

### The reusable timer module

At `/opt/vllm/vllm/v1/spec_decode/_mtp_timing.py`:

```python
class MTPTimer:
    def __init__(self):
        self.enabled = os.getenv("MTP_TIMING", "0") == "1"
        self.summary_every = int(os.getenv("MTP_TIMING_EVERY", "50"))
        self._events = defaultdict(list)   # label -> [(start_evt, end_evt), …]
        self._cpu_times = defaultdict(float)

    def enter(self, label):
        if not self.enabled: return
        start = torch.cuda.Event(enable_timing=True); start.record()
        t0 = time.perf_counter()
        self._open_stack.append((label, start, t0))

    def exit(self, label):
        if not self.enabled: return
        top_label, start, t0 = self._open_stack.pop()
        end = torch.cuda.Event(enable_timing=True); end.record()
        self._events[label].append((start, end))
        self._cpu_times[label] += time.perf_counter() - t0

    def tick(self, ctx=0):
        # every SUMMARY_EVERY iters, sync + summarise
        torch.cuda.synchronize()
        # print per-label: n, gpu_total, gpu_avg, cpu_total, cpu_avg
        …
```

Key design choices:

- **Env-var gated** (`MTP_TIMING=1`). Zero overhead when off (early `return`
  in every method). Default off → prod-safe.
- **`torch.cuda.Event` with `enable_timing=True`** for GPU time. Records
  asynchronously, no device sync at `.record()`. Syncs once per summary
  (`torch.cuda.synchronize()`) when flushing.
- **`time.perf_counter()` in parallel** for CPU time. A huge gap between
  "gpu_total" and "cpu_total" for the same label is the signal of a
  GPU→CPU sync — CPU is blocked waiting for the GPU to hand over a value.
- **Named labels** form a pseudo-tree (`P_build_total` contains
  `P_before_split`, `P_split_decodes_and_prefills`, `P00_prefill_branch_total`,
  `P00a_num_computed_tokens_cpu_path`, …). Lets you zoom in without
  reinstrumenting.
- **No per-iter sync** — only a single `torch.cuda.synchronize()` at
  summary flush. The flush happens once every `SUMMARY_EVERY` iterations
  (default 50) so instrumentation overhead is negligible during the run.

### How to actually use it (recipe)

1. **Apply the instrumentation patch** (example:
   `patches/mtp_sync_fix_and_instrumentation.patch`) which wraps sections of
   the metadata builder / scheduler / eagle proposer with `_t_enter(label)`
   / `_t_exit(label)` calls. The key feature of the patch is a `try:
   from vllm.v1.spec_decode._mtp_timing import enter as _t_enter …` with a
   no-op fallback, so the source stays usable when the timing module is
   absent.

2. **Apply the timer module itself** (the `_mtp_timing.py` file above).

3. **Launch the server with** `MTP_TIMING=1 MTP_TIMING_EVERY=50`.

4. **Run a load** that exercises the suspected hot path (single long-ctx
   request if you're hunting single-stream perf; high-conc if concurrency).

5. **Read the periodic summary from stderr** every 50 iterations:
   ```
   [MTP_TIMING iter=50 ctx_max=30000] GPU and CPU (perf_counter) ms/iter:
     label                                     n  gpu_total  gpu_avg  cpu_total  cpu_avg
     P_build_total                            50     125.0      2.5    4200.0    84.0  ← !!
     P00a_num_computed_tokens_cpu_path        50      0.2       0.004  3998.0   79.96 ← !! cpu ≫ gpu
     P00_prefill_branch_total                 50     120.0      2.4    4180.0    83.6
     P_split_decodes_and_prefills             50       4.8      0.096     5.0     0.1
     P_before_split                           50       0.3      0.006     0.4     0.008
   ```

### Interpreting the summary

Three patterns appear:

- **`gpu_total ≈ cpu_total`** → label is a pure-CPU Python block or a pure-GPU
  block. Fine. Either CPU-optimise or move on.
- **`gpu_total ≫ cpu_total`** → label launches GPU work that completes after
  CPU returns. Fine for CPU, but if the gpu_avg is big the GPU is the
  bottleneck for the label's block.
- **`cpu_total ≫ gpu_total`** *(the interesting one)* → CPU is blocked
  waiting for GPU. Almost always a `.cpu()` / `.item()` / `.tolist()` inside
  the label. Grep for those within the label's code range and you'll find
  the sync.

The Kimi discovery was exactly this: `P00a_num_computed_tokens_cpu_path`
showed `cpu_avg ~= 80 ms` while `gpu_avg ~= 0 ms` — 100 % CPU stall.
Following the label to the source pointed at:

```python
# in MLACommonMetadataBuilder.build(), prefill branch:
num_computed_tokens_cpu = (
    common_attn_metadata.compute_num_computed_tokens().cpu()  # ← the stall
)
```

The `.cpu()` forces the CPU to wait for all in-flight target-model GPU
work to finish before it can see the value, because `compute_num_computed_tokens`
was scheduled on the main stream after the last target-model launch. Under
`async_spec_decode` scheduling the optimistic CPU value was already
available — that's why upstream main simply skips this path with
`num_computed_tokens_cpu = None`.

### Why not nsys / NCU for this

- `nsys` would show the sync **in the timeline** but you'd need to zoom to
  ~0.1 ms granularity and correlate 61 layers × 100s of ops per decode
  step. The custom timer answers the question directly: "for label X, is
  CPU blocked on GPU?". Answered in one summary print.
- `NCU` is for **kernel-internal** perf (occupancy, memory bandwidth,
  compute saturation). It can't tell you that a Python-side `.cpu()` is
  stalling.
- The custom timer is **always-on-off at will** via env var, so you can
  enable it in a prod image, ship it, and turn it on only when needed.
  nsys requires a separate profiling run.

### When to reach for CUDA-event instrumentation

- You suspect a Python-side cost (metadata build, scheduler step) but can't
  see it in the kernel microbench.
- You suspect a GPU→CPU sync (`.cpu()` / `.item()`).
- You want per-branch attribution in a function with many early-returns and
  different code paths taken per workload shape.
- You want **live, cheap** perf measurement in production — CUDA events
  plus env-var gate have near-zero overhead off, and <1 % overhead on.

### When NOT to reach for it

- Kernel-internal attribution: use NCU or Triton's own autotuning traces.
- End-to-end tok/s comparisons: the e2e bench (§5) already gives that.
- Hot-path GPU-only sub-nanosecond timing: CUDA events have ~1 μs overhead;
  anything sub-μs needs different tooling.

### What was instrumented in this project

Two patch files capture the actual labels used, preserved for reference:

- [`patches/mla_eagle_instrumentation.patch`](patches/mla_eagle_instrumentation.patch)
  — labels inside `MLAAttention.forward` / `MLACommonImpl` + eagle proposer
- [`patches/mtp_sync_fix_and_instrumentation.patch`](patches/mtp_sync_fix_and_instrumentation.patch)
  — labels inside `MLACommonMetadataBuilder.build()` covering every branch
  (prefill / decode / spec-verify) with `P_build_total`,
  `P_before_split`, `P00_prefill_branch_total`,
  `P00a_num_computed_tokens_cpu_path`, …, `P99_after_metadata`

Apply either / both against a clean base (`kimi-k25-eagle3mla-current-20260423`)
to reproduce. The label names are stable so an agent can correlate their
own summary output against what the Kimi project found.

---

## 3. Kernel-level microbench (stand-alone, no vLLM server)

When you need to explore a config space beyond `num_kv_splits` (e.g.
`BLOCK_N`, `BLOCK_H`, `num_stages`, `num_warps`), don't touch the vLLM
server — it takes 30 s to start up per config. Write a stand-alone
microbench that calls the Triton kernel directly with synthetic tensors.

**Template**: see `bench/tune_triton_mla.py`. Key parts:

```python
from vllm.v1.attention.ops.triton_decode_attention import _fwd_grouped_kernel_stage1

# Synthesise tensors that match real production shapes
q = torch.randn(B, heads, LK, dtype=torch.bfloat16, device=device) * 0.1
kv = (torch.randn(pool_blocks, PAGE_SIZE, 1, LK, …) * 0.1).to(torch.float8_e4m3fn)
req_to_tokens = torch.randint(0, pool_blocks, (B, max_bpr), dtype=torch.int32, device=device)
b_seqlen = torch.full((B,), seq_len, dtype=torch.int32, device=device)
att_out = torch.empty(B, heads, num_kv_splits, LV + 1, dtype=torch.float32, device=device)

# One launch
def launch():
    _fwd_grouped_kernel_stage1[grid](
        q, kv, kv, sm_scale,
        req_to_tokens, b_seqlen, att_out,
        … strides …,
        …k/v scale …,
        kv_group_num=heads, q_head_num=heads,
        BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=512,
        BLOCK_N=..., BLOCK_H=..., NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=PAGE_SIZE, logit_cap=0.0,
        num_warps=num_warps, num_stages=num_stages,
        Lk=LK, Lv=LV, IS_MLA=True,
    )

# Time via CUDA events
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
for _ in range(WARMUP): launch()
torch.cuda.synchronize()
start.record()
for _ in range(TIMED): launch()
end.record(); end.synchronize()
per_call_ms = start.elapsed_time(end) / TIMED
```

### Pitfalls & lessons from the Kimi session

1. **Synthetic tensor dtype must match production.** Our first smoke test
   used `torch.uint8` for the fp8 kv pool and Triton errored with
   `only int8 supported!` inside the `tl.dot`. Fix: cast a bf16 tensor to
   `torch.float8_e4m3fn` (or `e4m3fnuz` on ROCm) — that sets the dtype tag
   the kernel needs.

2. **WARMUP + TIMED choice matters for wall-clock.** We started with
   WARMUP=3 + TIMED=20. At 54k total configs on 8 GPUs it was looking at
   4-hour completion. Dropped to WARMUP=2 + TIMED=5; 5 timed iters give
   <1 % variance on kernels > 1 ms — and the entire sweep fell to ~30 min
   wall-clock. For sub-ms kernels, bump TIMED back up. For a microbench
   that fires > 10k configs, think about budget first.

3. **Shmem pre-filter before the compile attempt.** sm120 dynamic shmem
   cap is 101 376 B. For `(BLOCK_N, BLOCK_H, num_stages)` tuples that
   exceed it, the Triton compile fails after ~200 ms of wasted work. Add
   an analytical formula up-front and skip:

   ```python
   def estimate_shmem(BLOCK_N, BLOCK_H, num_stages, LK=576, fp8_kv=True):
       k_tile = num_stages * BLOCK_N * LK             # fp8 = 1 B
       q_tile = BLOCK_H * LK * 2                      # bf16 = 2 B
       acc    = BLOCK_H * BLOCK_N * 4                 # fp32
       return k_tile + q_tile + acc + 4096            # +softmax scratch
   ```

   This cut the Kimi sweep by ~60 %.

4. **Parallelise across all GPUs with `CUDA_VISIBLE_DEVICES`.** The
   microbench is embarrassingly parallel over `(heads, max_model_len, B)`
   outer points. Round-robin assignment via `if i % world == rank` gives
   a clean even partition without any coordination. Write per-rank JSON
   outputs and aggregate at the end.

5. **Run on clean GPUs.** A live vLLM server holds ~90 GB per GPU; the
   microbench needs its own allocations and you'll hit OOM or silent
   contention. Always stop the server before sweeping.

### The single most important pitfall — seq_len bias in tuning

**Test at multiple seq_lens per `max_model_len` bucket, not one.** We made
this mistake twice in this project:

- **Round 1**: tested at `seq_len = 0.95 × max_model_len`. Geometric mean
  picks a config biased to the longest seq_len. At the deployment's real
  ctx=0 workload (actual seq_len ≈ 50–100 tokens) the config was wrong.
  Single-stream regressed 22 %.

- **Round 2**: added 4 fractions `(0.05, 0.30, 0.60, 0.95)`. Better, but
  still missed the "ctx=0" regime at mml=262 144 (5 % is still 13 k
  tokens). conc=128 ctx=0 stayed at −7 %.

- **Round 3 (final)**: 5 fractions `(0.01, 0.05, 0.30, 0.60, 0.95)`.
  Now covers everything down to seq_len = 2.6 k at mml=262 144 and 160
  at mml=16 000. Fixed the short-ctx cases; the 262 k bucket's
  `conc=128` winner still stayed at `splits=4` because geomean there is
  still dominated by the 95 % point, but all smaller buckets got
  better-balanced winners.

**Rule of thumb**: include at least one fraction ≤ 0.01 (1 %) for any mml
where the deployment will serve very short ctx (chat, code gen, anything
with short outputs at ctx=0). If your picker is geometric or arithmetic
mean, consider weighted means that prefer short seq_len for small-B
buckets — the heavy long-ctx numbers otherwise dominate.

---

## 4. A/B diagnostics when the microbench says "fine" but e2e says "slow"

Classic case: kernel microbench shows config X is N % faster than config Y,
but the e2e deployment shows X is slower. Means the delta is
**out-of-kernel**: Python-side prep, memcpy, CG pool pressure, metadata
build, launch overhead.

### Override test — attribute to kernel config

To test whether **kernel config is the limiting factor**, manually override
the tuning table for one bucket to the suspected-better config and re-run
the e2e bench:

```python
# In triton_mla_tuning.py:
TUNED_KV_CONFIGS = {
    ...
    (128, 262144, 128): {"num_kv_splits": 1, "BLOCK_N": 32, "BLOCK_H": 8,
                         "num_stages": 2, "num_warps": 4},  # HAND-SET, what
                                                            # we think should win
}
```

Deploy, re-bench. Interpret:

- **Performance changes toward the manually-set config's predicted value**
  → kernel config is in the critical path. Tune more.
- **Performance doesn't change or moves wrong way** → kernel config is not
  the bottleneck. Look elsewhere.

In the Kimi session, the override test eliminated kernel config as the
explanation for a 130-tok/s delta. That saved us from a couple more rounds
of futile tuning.

### `.zero_()` / memcpy hunt

For per-step overheads (things that scale with decode step count, not
request count), grep for any `.zero_()`, `.copy_()`, `.fill_()`,
`.clone()`, `.cuda()`, `torch.empty`, `torch.zeros` in the hot-path
metadata builder:

```bash
grep -n "zero_\|\.copy_(\|\.clone(\|\.fill_(\|torch\.empty\|torch\.zeros" \
  /opt/vllm/vllm/v1/attention/backends/mla/triton_mla.py \
  /opt/vllm/vllm/model_executor/layers/attention/mla_attention.py
```

For each hit, ask:

- Does it run every step or one-time at init?
- What's its byte size at the CG-captured max? (scale with `max_num_seqs`,
  `max_model_len`, etc.)
- Is it actually required, or defensive?

In the Kimi session, one such hit — a `.zero_()` on the tail of the
`cg_buf_block_table` each step — turned out to cost ~21 % on MTP conc=128.
The replacement was trivial (`torch.empty → torch.zeros` at init, drop the
per-step zero_). Low-hanging fruit worth the grep.

---

## 5. CLI isolation — when the kernel is innocent

If kernel override didn't move the needle, the remaining suspect is the
**CUDA-graph capture shape / deployment CLI**. CLI flags that affect CG
shapes at capture time:

- `--max-model-len` → sets `max_blocks_per_req = cdiv(mml, block_size)` and
  thus the width of every CG-captured `block_table` and stride layout.
- `--max-num-seqs` → sets CG-captured batch dimension.
- `--max-num-batched-tokens` → affects scheduler chunk sizes, not decode
  directly but affects interleaving.
- `--speculative-config num_speculative_tokens` → changes the CG-captured
  number of tokens per request in spec-verify.

**Isolation protocol**: change exactly **one** flag per run, start from a
single baseline, and bench. If changing flag X alone reproduces the known
target throughput, X accounts for the full delta. In the Kimi session:

| variant | changed from baseline | conc=128 ctx=0 |
|---|---|---:|
| baseline | — | 1397 |
| + `--max-model-len 150000` only | 1 flag | 1527 |
| + `--max-num-batched-tokens 4096` only | 1 flag | 1397 (hyp., skipped) |
| + `--language-model-only` only | 1 flag | 1397 (hyp., skipped) |

Once one flag alone reproduces the target, the other flags can be skipped
(saves ~15 min each). But **do not skip** if you need to rule out a
second-order interaction — do all three if conservative.

**Why `--max-model-len` matters at decode time**: CG captures `block_table`
at `max_num_seqs × cdiv(max_model_len, block_size) × int32`. At
`mml=262144, block_size=16, max_num_seqs=128` that's 8.4 MB with a 64 KB
row stride. The kernel scans `block_table[req, 0:actual_bpr]` with that
wide stride even if only the first few entries matter at ctx=0. Wide stride
→ cache / prefetcher pressure across 61 layers × many steps/sec. The
microbench doesn't see this because it allocates block_table at the actual
bpr, not the captured max.

**Takeaway**: the microbench is a **kernel-level** measurement. CG-level
costs are not captured there. Only e2e bench with matching CLI will expose
them.

---

## 6. Workflow summary (do this top-to-bottom on a new problem)

```
┌─ STEP 1. Log read
│    grep "cudagraph_mode\|CUDA graph pool\|Traceback\|CompilationError" <log>
│    expect: FULL cudagraph for every hot-path backend, no CompilationError
│
├─ STEP 2. Analytical CG + grid
│    For each captured (B, heads) bucket:
│      - grid = (B, ceil(heads/BLOCK_H), NUM_KV_SPLITS)
│      - ratio = grid / SM_count    target: 2-8
│    If ratio > 8 or < 0.5 for any bucket → fix constexpr-per-bucket (§2)
│
├─ STEP 2.5. CUDA-event branch instrumentation (live server, §2.5)
│    Apply mtp_sync_fix_and_instrumentation.patch + _mtp_timing.py
│    Launch with MTP_TIMING=1 MTP_TIMING_EVERY=50
│    Look for cpu_avg ≫ gpu_avg in the summary: that's a GPU→CPU sync
│    (How the 40-80 ms .cpu() stall was found.)
│
├─ STEP 3. Kernel microbench (stand-alone)
│    bench/tune_triton_mla.py --rank 0 --world 8 --out tune_gpu0.json
│    (parallelise across GPUs; pre-filter shmem; sweep at MULTIPLE seq_len)
│    See "single most important pitfall" in §3.
│
├─ STEP 4. Override diagnostic
│    Patch the tuning table for ONE bucket to a known-good config.
│    Re-bench e2e. If it moves → kernel. If not → not kernel.
│
├─ STEP 5. Per-step overhead grep
│    grep "zero_\|\.copy_(\|\.fill_(\|\.clone(" in hot-path builders
│    For each, estimate bytes × steps/sec. Nuke the unnecessary ones.
│
└─ STEP 6. CLI isolation
     Change one flag at a time, re-bench e2e. Attribute the gap.
```

Each step has a clear YES/NO exit criterion and hands off to the next step
only if the answer is "not here".

---

## 7. What NOT to do

- Do **not** profile with nsys-systems or NCU as your **first** tool. NCU
  on a 61-layer model × all buckets gives ~40 GB of traces and a week of
  analysis. Use it **only** when you've narrowed the problem to a single
  kernel and need cycle-level attribution. **Do** use CUDA-event branch
  instrumentation (§2.5) — it's much more targeted and can stay in a prod
  image behind an env var.
- Do **not** launch the vLLM server inside your sweep loop. 30 s startup ×
  10k configs = forever.
- Do **not** commit an un-guarded hardware-specific tuning table
  (`TUNED_KV_CONFIGS`) upstream. Use analytical fallback + script-only
  upstream path. (This is called out in
  [`upstream-pr-plan-review.md`](upstream-pr-plan-review.md).)
- Do **not** trust "it was fine before we touched it". The Kimi session's
  Phase 3 regression was from a constant that was correct for single-req
  long ctx but wrong for high concurrency; nothing about it was "broken"
  in isolation.
- Do **not** optimise by comparing to a single baseline number. The user's
  1508 baseline ran with a **different CLI** than the post-session runs.
  Matching the number required an independent CLI isolation test, not
  more kernel tuning. See §5.

---

## 8. Tools used in this session, mapped to the workflow

| tool | used in | notes |
|---|---|---|
| `grep` on `/opt/vllm/vllm/**` source | §1, §5 | find builder classes, `_cudagraph_support`, `zero_` etc. |
| `nvidia-smi --query-gpu=…` | any | quick GPU-is-free check before microbench |
| **`/opt/vllm/vllm/v1/spec_decode/_mtp_timing.py` + `MTP_TIMING=1`** | **§2.5** | **branch-level timing in live server — how the `.cpu()` sync was found** |
| `torch.cuda.Event` | §2.5, §3 | GPU timing both in the custom timer (live server) and the microbench (stand-alone) |
| `CUDA_VISIBLE_DEVICES=k python …` | §3 | per-GPU parallel microbench |
| `docker commit && docker push` | §4–5 | ship an image to test on another box or roll back |
| `docker cp` + file re-deploy | §4 | iterate fast on `triton_mla.py` inside a running image |
| `/mnt/llm_decode_bench.py` | §4, §5 | end-to-end throughput probe (user-supplied) |
| `bench/e2e_bench.py` | §1 (original) | streaming tok/s + interarrival probe |

---

## 9. Open loose ends for the next person

- Multi-bucket CG capture for `max_model_len` would close the §5 trade-off
  (short-ctx speed vs long-ctx flex) without forcing the user to pick.
  Non-trivial change to vLLM core; punted.
- Adaptive MTP kill-switch based on running batch size (not implemented
  here).
- Stage-2 merge kernel (`_fwd_kernel_stage2`) wasn't swept — its defaults
  (`num_warps=4, num_stages=2`) are probably fine but a quick sweep could
  verify.
- Integration test for the FULL-CG spec-verify path is missing; upstream
  review will ask for one.

---

*File maintained at
`/root/rtx6kpro/models/kimi-k26-mtp-long-ctx-wip/debugging-methodology.md`.
Session date: 2026-04-23 / 2026-04-24.*
