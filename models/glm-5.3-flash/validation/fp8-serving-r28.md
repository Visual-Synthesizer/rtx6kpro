# GLM FP8 serving and checkpoint qualification

Status: **qualified for the contracts below**, measured on 2026-09-08.
This historical report records the R28 artifact and its R27 comparison for the
[GLM deployment specification](../../glm-5.3-flash.md). It does not claim every workload is
faster, universal bitwise generation parity, Qwen qualification, or NVFP4
target-KV qualification.

## Artifact boundaries

The published image is
`voipmonitor/vllm:jovian-judgement-community-20260908-r28`, with registry digest
`sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669`
and image ID
`sha256:24b06eacca12e16cd94a9ed3b1987f5ef283b6358d5792664dcdd0ceaf7fdb17`.
Its [source lock](fp8-serving-r28.source.lock) identifies committed vLLM,
B12X and LMCache trees, native artifacts and launchers. Registry verification
confirms the image ID and two filesystem layers.

Performance qualification composes evidence across explicitly identified
artifacts, not across inferred tag names:

| Role | Image ID | Coverage |
|---|---|---|
| Published R27 reference | `aef5832d6661375dfc94607ae89db8c99af50c015010f61c12d06d958fbb53d1` | Same-quartet serving comparisons and complete-output MTP reference |
| Disjoint-MLA serving kernels | `8bac8633684bfd0aa2f49a33acc2880a589ac1f7372a7a6913ae2202011881f2` | No-spec/DFlash duration comparisons; six mode/DCP million-token cache contracts |
| Tool history and retrieve completion | `9bbd669ef6a5e3a9fbfcf842004078f1ae167e88968a159697ae32bda89797bf` | MTP duration comparisons, tool calling and MTP external-restore sampling |
| Evictable semantic payloads | `d1be0f34bedfd920e1d27aaa16441fe74492898d3d86d7d70ff6514c3c41c2b9` | DFlash restore sampling and fixed-execution numerical controls |
| Published package | `24b06eacca12e16cd94a9ed3b1987f5ef283b6358d5792664dcdd0ceaf7fdb17` | Installed-package tests, DFlash/DCP4 storage smoke and complete-output MTP/DCP4 comparison |

The serving-kernel composition uses vLLM `9aa73287b12ee76114105e6fb838003e62b6d030`.
The tool and publication compositions use vLLM
`2531689fa50b956d3e1156e1ab80d119aaf34c1e`; their vLLM delta handles tool history
and asynchronous completion, without changing model kernels. Every image in
the four non-reference rows uses B12X
`3edbcbce70f491741b82f5eab9c1b30b39447228`. The LMCache capacity and event-setup
changes have separate exact-package checks; the publication does not present
the entire duration matrix as a rerun on the final image ID.

## Serving rates

All comparisons use stock RTX PRO 6000 Blackwell Workstation Edition GPUs,
TP4, FP8 target KV, scheduler budget 4096, OMP1, 16 NCCL channels, 2 MiB buffers
and full-and-piecewise CUDA graphs. A/B uses the same physical quartet for
each mode. GPUs 0–3 serve no-spec, 4–7 MTP3 and 8–11 DFlash2. Absolute rates
across quartets are not an isolated speculation-mode comparison.

Each duration cell contains three warmed 30-second samples; MTP3/DCP4 and
DFlash2/DCP1 retain six samples from both boot orders.

| Mode | DCP | 32K prefill, R27 → R28 tok/s | C1 output, R27 → R28 tok/s | C8 total output, R27 → R28 tok/s |
|---|---:|---:|---:|---:|
| No-spec | 1 | 14,773 → 14,692 (−0.55%) | 158.80 → 158.86 (+0.04%) | 704.42 → 701.11 (−0.47%) |
| MTP3 | 1 | 14,428 → 14,378 (−0.35%) | 249.73 → 247.55 (−0.87%) | 885.30 → 892.23 (+0.78%) |
| DFlash2 K7 | 1 | 14,318 → 14,252 (−0.46%) | 208.97 → 206.90 (−0.99%) | 688.13 → 681.52 (−0.96%) |
| No-spec | 4 | 13,180 → 13,531 (+2.66%) | 141.80 → 142.21 (+0.29%) | 647.73 → 639.71 (−1.24%) |
| MTP3 | 4 | 12,890 → 13,195 (+2.37%) | 233.43 → 225.25 (−3.50%) | 819.19 → 829.69 (+1.28%) |
| DFlash2 K7 | 4 | 12,814 → 13,102 (+2.25%) | 187.84 → 196.14 (+4.42%) | 627.93 → 629.43 (+0.24%) |

| Mode | DCP | C1 verifier, R27 → R28 steps/s | C8 aggregate verifier, R27 → R28 steps/s |
|---|---:|---:|---:|
| MTP3 | 1 | 102.64 → 102.85 (+0.20%) | 365.80 → 365.93 (+0.04%) |
| MTP3 | 4 | 91.71 → 92.20 (+0.53%) | 334.02 → 337.57 (+1.06%) |
| DFlash2 K7 | 1 | 85.54 → 85.29 (−0.29%) | 285.26 → 283.23 (−0.71%) |
| DFlash2 K7 | 4 | 77.99 → 78.16 (+0.22%) | 256.14 → 258.77 (+1.03%) |

Prefill is input tokens divided by API TTFT, including first-output work.
Speculative output also depends on proposal acceptance; aggregate verifier
rates sum request progress rather than physical batched graph launches.
The MTP3/DCP4 C1 output cell is **233.43 → 225.25 tok/s
(−3.50%)**, outside the 2% median-loss gate, despite **91.71 → 92.20 steps/s
(+0.53%)**. No observation from that cell is discarded.

The [complete-output MTP comparison](mtp3-dcp4-complete-output-r28.json) fixes
24 seeds per image and 4096 output tokens per request, temperature 1 and
top-p 1, using the same 78-token mathematics prompt as the duration benchmark.
There are 96 measured requests, with alternating fresh/repeated order.

- Fresh output: **228.35 → 232.97 tok/s (+2.02%)**; verifier **+0.59%**.
- Repeated output: **229.43 → 231.64 tok/s (+0.96%)**; verifier **+0.55%**.
- Within R28, repeated versus fresh: output **−0.57%**, verifier **−0.04%**.
- R27 repeats compute all 78 prompt tokens; R28 repeats restore all 78 and
  compute zero. All requests complete without preemption or external transfer.
- All 2247 clock samples have zero offsets. All 2251 PCIe samples contain no
  degraded active link on GPUs 4–7.

The declared complete-output median gates pass. The paired geometric
checkpoint output interval is −1.62% to +1.30%. This control does not reproduce
a persistent cache penalty; it does not erase the failed duration result,
prove a speedup for every prompt, or establish strict statistical
non-inferiority. The seed count was fixed before candidate sampling.

## Checkpoint storage

No-spec, MTP3 and MXFP8 DFlash2 each pass TP4/DCP1 and TP4/DCP4 cold, local
prefix-cache, RAM restore and filesystem restore contracts. Each million-token
external restore attributes 1,000,000 prompt tokens to storage and zero to
compute. API restore times are 0.694–0.996 seconds from RAM and 0.977–3.157
seconds after restarting both services. The OS filesystem cache was not
flushed; these are not cold-device I/O measurements.

| Mode | DCP | 1M cold, seconds | RAM restore, seconds | Restore after service restart, seconds |
|---|---:|---:|---:|---:|
| No-spec | 1 | 93.899 | 0.918 | 1.768 |
| No-spec | 4 | 93.905 | 0.694 | 1.147 |
| MTP3 | 1 | 97.102 | 0.900 | 3.157 |
| MTP3 | 4 | 97.739 | 0.739 | 0.977 |
| DFlash2 | 1 | 95.957 | 0.996 | 1.618 |
| DFlash2 | 4 | 96.374 | 0.722 | 0.987 |

The [exact-package storage result](packaged-checkpoints-r28.json) passes:

- Literal 54,643-token lookup answers across cold, local, RAM, filesystem and
  restart paths; shared leading SYSTEM reuse and changed-SYSTEM misses.
- C4 transfer byte identity on every rank.
- Three C8 cancellation/live-read eviction rounds: 24 generations, 3456
  transfers and 21,838,823,424 verified bytes.
- 58 installed connector, storage and index contract tests.

Capacity tests force a 64 GiB RAM pool through five eviction cycles and 151
semantic generations, with zero remaining leases or pending tasks. Version-2
rank/group storage keys are eligible for ordinary LRU eviction while complete
generation manifests retain atomic retrieval. Version-1 semantic payloads
produce safe misses rather than partial restores.

## Sampling and numerical scope

MTP3/DCP1 retains 24 seeds and 72 cold-A/cold-B/RAM requests. Cold-A/RAM output
is 312.01/310.34 tok/s (−0.53%); verifier rate changes −0.12%.
The [DFlash2/DCP1 sampling result](dflash2-dcp1-restore-sampling-r28.json)
retains all 48 seeds and 144 requests: cold-A/RAM output is 386.14/381.20 tok/s
(−1.28%); verifier changes −0.14%. Its first 24-seed cohort failed the median
output gate; the retained 24-seed extension is exploratory, not a predeclared
statistical non-inferiority proof. No further cohort was added.

Diagnostic DFlash controls fix MoE execution and canonical cache addressing.
For two seeds with 4096 outputs, cold and restored token IDs, reported top-10
log probabilities and three observed full-vocabulary proposal tensors are
bit-identical. Those diagnostic controls are not production speed settings.

A separate DFlash/DCP4 continuation differs between cached and bulk prefill.
A cold-only split at token 33,023 of a 33,029-token prompt reproduces the same
32-token continuation with zero cache hits. All-rank storage bytes match
independently. Different floating-point prefill partitions therefore are not
a universal bitwise generation oracle; storage-byte correctness does not by
itself prove distribution or task-quality equivalence.

Fine-aligned MTP3/DCP4 compares 256-token versus 2048-token recurrent retention
while keeping 2048-token attention pages: prefill −0.20%, C1 verifier +0.32%
and C8 verifier +0.35%. That interval comparison predates the disjoint-MLA
projection change and is not a fresh performance claim for every mode.

## Historical release changes: R27 to R28

- Extends exact request and leading SYSTEM/developer checkpoint reuse to
  DFlash2 and DCP4, alongside no-speculation and MTP3. Different user
  continuations can reuse their shared instruction prefix in all six modes.
- LMCache stores those semantic checkpoints as immutable, all-rank bundles.
  Worker-owned asynchronous SHM copies support RAM and filesystem restore,
  including restart of both services, without a sidecar CUDA context.
- Cancellation and eviction cannot recycle checkpoint pages while copies are
  using them. Full RAM pools reclaim eligible payloads; event setup and copy
  submission failures release task ownership correctly.
- Fine aligned retention can keep recurrent state every 256 tokens while
  attention pages remain 2048 tokens. Packed prefill exports avoid a separate
  target forward at every recurrent checkpoint, and speculative convolution
  history remains consistent with the accepted prefix.
- GLM DCP sparse-index compaction preserves selected-index order and initializes
  counts and invalid tails without separate fill kernels.
- Shared-expert output retains its allocation until consumer-stream reads finish.
  Disjoint MLA projection batches avoid the documented SM120/121 cuBLAS
  allocation-boundary fault while retaining CUDA graphs.
- Truncated tool-call arguments remain parseable; malformed tool history does
  not crash template rendering. Aborted external retrieves cannot complete
  twice or free unrelated active requests.
- Qwen source integration is included for independent testing. No Qwen model
  execution or performance qualification is claimed for this release.

## Review and reproduction

The [source-locked build recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
and [merge checklist](https://github.com/local-inference-lab/vllm/issues/651)
identify the complete sources and required integration resolutions. Original
authors remain recorded in the published Git mirrors. Changing the runtime,
model revision or cache identity requires its own qualification.
