# GLM FP8 scheduler and launcher qualification

Status: **qualified for the contracts below, with the stated performance limits**.
This report evaluates
the scheduler-only GLM-5.3-Flash serving artifact against its immutable R28
parent. It does not qualify a replacement model kernel, NVFP4 target KV,
pipeline parallelism, TP8, Qwen execution, or a cn4 deployment.

The [raw samples and hardware gates](scheduler-serving-r28.1.json) retain
short and extended cohorts separately. The [source lock](scheduler-serving-r28.1.source.lock)
identifies the image's complete component and launcher inputs. The
[R28 historical report](fp8-serving-r28.md) records the inherited six-mode
checkpoint and numerical qualification.

## Artifact contract

The published scheduler image is
`voipmonitor/vllm:jovian-judgement-community-20260908-r28.1`, registry digest
`sha256:52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f`.

The parent is `voipmonitor/vllm:jovian-judgement-community-20260908-r28`,
registry digest
`sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669`.
The scheduler artifact has image ID
`sha256:259a592cc1b86312a9bb61bd95c1aad66577d468a5502b4191ec69e22313fede`.
It contains three filesystem layers: the parent's two layers and one
Python/launcher overlay, not a chain of intermediate qualification images.

The complete vLLM source is
`9ff42d83938e74018f9c255e8cfa7ca6df6921b0`, tree
`f3651af6738fc4b425208a428c7bc4ff9f334f35`. The source-lock SHA-256 is
`4473b46dbf696a386da1fbd6f75e7ef9159c36d216d153c6beb5cfe68b7a7477`.
Installed version metadata agrees with this Python source identity.
B12X, LMCache, FlashKDA, CUDA, PyTorch and NCCL are unchanged. Inspection
verifies all fourteen vLLM/LMCache shared libraries are byte-identical to R28,
along with the parent layer identities and clean installed Git trees.

## Historical release delta: R28 to R28.1

- Contended compute-share timing uses a primitive timestamp paired with its
  executor batch. It no longer allocates a timing helper or attaches a Python
  callback to the model future. Untimed predecessor completion is observed
  only when an already-dispatched timed successor needs its queue boundary.
- Uncontended decode avoids the additional runnable-decode count and prefill
  interleaving entry. Automatic demand observation remains separate; this is
  not a claim that every scheduler operation is constant-time.
- `max_parallel_prefills=auto` resolves to `min(4, max_num_seqs)`, independently
  of cache page/object size and the global token budget. Request priority,
  work-conserving service and isolated boundary-logits restore are preserved.
- The launcher accepts all five scheduler environment controls, including
  automatic share and its half-life. Explicit native CLI arguments win without
  constructing duplicate flags. Fixed share 0.4 and one lane remain defaults.
- The GLM launcher's chat reasoning default is `high`. Explicit `max` and
  `low` requests override it. Direct `vllm serve` bypasses launcher defaults.

The canonical source reviews are vLLM #664 and blackwell-llm-docker #31.
Derek Yates's hardening contribution retains authorship. The complete serving
mirror preserves the checkpoint integration resolutions needed by the image.

## CPU and artifact evidence

The installed serving image and complete source tree each pass 159 focused
tests. The canonical #664 branch merged with its JJ base passes 156 focused
tests. Launcher and source-installation contracts pass 43 tests. An independent
installed-image replay of the five fairness suites passes 135 tests with
read-only test dependencies and cached CLI model metadata; no model executes
in that CPU-only replay.

Coverage includes disabled timing without clock reads, deferred sampling,
exact batch/timestamp pairing, external transfers, cache-geometry-independent
lanes, request priority and boundary isolation across disabled/fixed/automatic
share and round-robin/decode-aware admission. Launcher tests cover invalid
values, CLI precedence, half-life rules and help without model initialization.
Live tokenization confirms default reasoning equals explicit high and that
explicit max/low change the rendered prompt.

A minimal two-slot engine reproducer dispatches a timed successor at 10.1 s,
observes its untimed predecessor at 10.2 s, and completes it at 10.5 s. Charging
0.4 s includes predecessor residency; the corrected service is 0.3 s. Eight
tests cover transfer/model predecessors, timed/untimed successors and deferred
sampling. Wholly untimed queues still read no clock and attach no callback.
This is engine-observed service time, not CUDA-event measurement of kernels.

## Performance method

Every performance ratio uses physical GPUs 0–3: four stock RTX PRO 6000
Blackwell Workstation Edition cards, active PCIe Gen5 x16. GPU clocks and
offsets are recorded throughout; GPU4–7's independently overclocked service
does not supply these measurements. Concurrent cache validation on GPUs8–11
is serialized outside primary benchmark execution.

The MTP configuration is TP4/DCP4, FP8 target KV, MTP3, engine-driven LMCache,
64 maximum sequences, fixed prefill share 0.4 and one round-robin lane. Both
images use a 4096-token scheduler budget, OMP1, NCCL16 with 2 MiB buffers,
FULL_AND_PIECEWISE graphs, maximum capture size256 and a 64 GiB CPU cache.
Startup reports 14,746,058 usable model KV tokens in both images.

The pinned `llm_decode_bench` 0.4.29 source has SHA-256
`a17ee69dd2ee5aa59d9c9a1b03e28cae6fe2837545ecc967256b2828215deab7`.
Both arms use reasoning **max** during decode, so the launcher's changed
default cannot be mistaken for a scheduler speedup. Each concurrency is
warmed before its 30-second cell. C64 has three additional 60-second repeats;
C1 has separately recorded longer repeats. Reported verifier steps sum request
progress, not physical batched CUDA graph launches.

Nominal 32K prefill uses a cold unique context bucket: approximately
32,315 actual prompt tokens. A discarded 30-second warmup precedes two
30-second samples. API input-tokens/TTFT includes first-output work and is not
an isolated GPU prefill timer. An independent token-ID probe submits exactly
204,800 tokens, verifies zero cache hits and records all computed tokens. The
benchmark's internal 131,072-token cap is not used as 200K evidence.

The first short C1 cell records 244.325 to 236.173 output tok/s (-3.34%),
97.290 to 97.542 verifier steps/s (+0.26%), and accepted length 2.512 to 2.422.
That observation is retained. Three 60-second C1 repeats per image, including
a reverse-boot R28 control, record medians 231.649 to 238.366 output tok/s
(+2.90%) and 97.803 to 97.556 verifier steps/s (-0.25%). Output ranges overlap:
229.758–244.447 for R28 and 233.947–241.691 for the scheduler artifact. The
short output drop does not persist in these samples. Neither stable verifier
speed nor a changed acceptance sample proves distribution equivalence or a
universal speedup.

### MTP3/DCP4 measurements

One-lane controls, R28 → R28.1:

| Measurement | Output or prefill tok/s | Verifier request-steps/s |
|---|---:|---:|
| C1, median of three 60-second cells | 231.649 → 238.366 (+2.90%) | 97.803 → 97.556 (−0.25%) |
| C64, median of three 60-second cells | 2153.662 → 2155.564 (+0.09%) | 844.671 → 846.087 (+0.17%) |
| Cold nominal 32K, median of two cells | 13141.5 → 13117.0 (−0.19%) | — |
| Exact 204,800-token cold input | 12773.973 → 12738.596 (−0.28%) | — |

The 204,800-token R28 control is one observation. The candidate has two
observations, 12730.256 and 12746.936 tok/s. These and the sub-percent decode
changes do not establish a general speedup. All measured decode requests
complete without errors, underfill, or capacity-limit flags.

The six-concurrency 30-second sweep is retained separately from the longer
C1/C64 controls:

| Concurrency | R28 output tok/s | R28.1 output tok/s | R28 verifier steps/s | R28.1 verifier steps/s |
|---:|---:|---:|---:|---:|
| 1 | 244.325 | 236.173 | 97.290 | 97.542 |
| 4 | 553.244 | 566.738 | 232.496 | 230.549 |
| 8 | 862.372 | 872.463 | 344.722 | 343.999 |
| 16 | 1232.698 | 1237.365 | 489.957 | 491.504 |
| 32 | 1688.849 | 1678.246 | 654.687 | 655.772 |
| 64 | 2254.639 | 2259.338 | 864.846 | 860.624 |

### No-speculation and DFlash2/DCP1 controls

These use GPU-local cache, 32 maximum sequences and the same physical GPUs0–3,
4096-token budget and explicit max reasoning. There is one warmed 30-second
decode cell per concurrency and two 30-second prefill cells per image.
No-speculation uses R28 then R28.1; DFlash2 reverses that image order.

| Mode | Measurement | R28 → R28.1 tok/s | Verifier request-steps/s |
|---|---|---:|---:|
| No-spec | C1 | 158.598 → 158.754 (+0.10%) | — |
| No-spec | C8 aggregate | 696.627 → 700.332 (+0.53%) | — |
| No-spec | Nominal 32K prefill | 14703.0 → 14708.5 (+0.04%) | — |
| DFlash2 K7 | C1 | 194.313 → 203.170 (+4.56%) | 91.022 → 90.535 (−0.53%) |
| DFlash2 K7 | C8 aggregate | 709.101 → 692.155 (−2.39%) | 293.555 → 290.677 (−0.98%) |
| DFlash2 K7 | Nominal 32K prefill | 14556.5 → 14559.5 (+0.02%) | — |

**DFlash C8 is an unresolved short-duration observation, not an equivalence
claim.** Accepted length changes 2.416 → 2.381 (−1.42%) alongside the verifier
difference. Additional extended C8 controls were not run after the release
owner requested publication without further repetitions. The image changes
scheduler/launcher Python, not draft sampling, model kernels or native
libraries, but source scope alone does not prove that a measured difference is
noise. This release does not claim every decode cell is non-regressing.

Startup KV capacity is 4,686,328 tokens for both no-spec controls. DFlash
reports 3,691,743 → 3,676,628 tokens under automatic memory sizing (−0.41%).
These are observed boot capacities, not a change to the allocation algorithm.

## Cache and fairness gates

**Qualified:** the four-lane MTP3/DCP4 FP8 configuration retains exact
input/SYSTEM boundary reuse, million-token RAM and restart-filesystem restore,
literal answer correctness, all-rank byte integrity and cancellation/read-lock
ownership. Its development endpoint reports four lanes, refill target4,
decode-aware admission and fixed share0.4 from environment settings alone.
The LMCache sidecar remains CPU-only. Filesystem page cache is not flushed;
restart latency is not a cold-device storage benchmark.

The exact candidate, stock GPUs8–11 and a separate API/cache namespace, passes:

- One million cold prompt tokens in 99.284 s; RAM restore in 0.855 s; restore
  after restarting both services in 0.970 s. Both restores attribute all
  1,000,000 tokens to external storage, zero to local compute, and return the
  same greedy output as cold execution.
- A 54,641-token literal lookup across cold, GPU prefix cache, RAM,
  filesystem and restart paths. Different user turns reuse an 11,340-token
  SYSTEM prefix; a changed SYSTEM prompt correctly misses.
- C4 all-rank byte identity and three C8 cancellation/live-read eviction
  rounds: 24 generations, 2304 verified transfers and 15,882,780,672 bytes.
- Four installed LMCache RAM-pressure, read-lock and cancellation tests.

A separate same-quartet, one-observation RAM comparison records 0.725 s for
R28 and 0.813 s for R28.1. Both restore all one million tokens without
recomputation. The 88 ms difference is retained; these single observations
are insufficient to establish a steady-state transfer-bandwidth regression
or improvement. Cache correctness and transfer speed are separate claims.

One-lane and four-lane mixed traffic are compared separately from pure decode.
Eight cold32K prefills precede a late4K request while two or four decodes run.
With four active decodes and refill target4, nearest-prefill promotion need
not activate; the two-decode workload exercises a depleted decode reservoir.
Three repetitions per configuration produce:

| Active decodes | Late 4K TTFT, one lane → four lanes | Median long-request TTFT, one lane → four lanes |
|---:|---:|---:|
| 4 | 46.850 → 8.726 s | approximately 26.1 → 47.3 s |
| 2 | 46.844 → 2.815 s | approximately 26.1 → 43.3 s |

All eight long prefills and the late short prefill finish in every run.
The result is a latency trade-off: a short request no longer waits behind
all long requests, while those long requests share service and often receive
their first token later. It is not a universal throughput improvement.
With four active decodes, long-request maximum TTFT changes from about
46.6 s to 48.5 s. The longest observed decode gap is 0.695 s with four lanes,
versus 0.349 s with one lane; these are finite-run observations, not latency
guarantees. One lane remains the image default.

Pure C64 decode, three 60-second repetitions on the candidate, is essentially
unchanged by selecting four lanes: median output 2155.564 → 2165.014 tok/s
(+0.44%), verifier 846.087 → 847.030 steps/s (+0.11%).

A single-decode collision with a cold 65,535-token prefill realizes a median
prefill compute share of 0.421 under fixed target0.4. R28 → R28.1 median
prefill is 5632.72 → 5723.20 tok/s, and decode verifier progress during the
collision is 54.578 → 55.018 steps/s. The first candidate collision includes
logged JIT compilation of KDA/cache-gather signatures and a 1.080 s maximum
decode gap; the two repeats record 0.313 and 0.314 s. The cold-signature
observation remains in the evidence. Startup did not warm every mixed-workload
kernel signature, and this scheduler update does not claim otherwise.

The unchanged R28 checkpoint/kernel composition has separate all-six
TP4 mode/DCP qualification. Focused R28.1 reruns do not imply the entire
R28 numerical study was repeated on this scheduler overlay.

The exact R28.1 GPU-local DFlash2/DCP1 aligned-256 control restores 8960 and
17152 token prefixes from inside 2048-token attention pages. Separately salted
cold controls report zero hits. All restored, cold and repeated literal
document answers are exact. This qualifies internal checkpoint reuse, not a
new aligned-mode performance or universal token-parity claim.
