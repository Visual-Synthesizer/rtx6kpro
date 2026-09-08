# GLM FP8 serving and checkpoint qualification

Status: **qualified for the contracts below**, measured on 2026-09-08.
This report supports the [GLM deployment specification](../../glm-5.3-flash.md)
and its R27-to-R28 performance table. It does not claim every workload is
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

The deployment page retains all six 32K/C1/C8 rows. Each duration cell contains
three warmed 30-second samples; MTP3/DCP4 and DFlash2/DCP1 retain six samples
from both boot orders. The MTP3/DCP4 C1 output cell is **233.43 → 225.25 tok/s
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

## Review and reproduction

The [source-locked build recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
and [merge checklist](https://github.com/local-inference-lab/vllm/issues/651)
identify the complete sources and required integration resolutions. Original
authors remain recorded in the published Git mirrors. Changing the runtime,
model revision or cache identity requires its own qualification.
