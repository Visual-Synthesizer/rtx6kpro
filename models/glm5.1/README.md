# GLM-5.1 on RTX PRO 6000 Blackwell

This section tracks GLM-5.1 experiments on RTX PRO 6000 Blackwell PCIe systems.

## Reports

| Date | Report | Scope |
|---|---|---|
| 2026-06-07 | [GLM-5.1 v9 BF16 TP16 DCP8 16-GPU runbook](../glm5.1_v9.md) | Exact Docker image, model path, 16-GPU TP16/DCP8 launch command, KV-cache sizing, required shared-experts overlay, startup log signatures, smoke tests, and known bad memory settings for the BF16 GLM-5.1 runtime. |
| 2026-05-25 | [GLM-5.1 teacher-forced decode KLD / JS reproduction](glm51-teacher-forced-decode-kld-js-2026-05-25.md) | Reproducible BF16-reference decode KLD/JS workflow, Hugging Face reference logits, helper scripts, exact single-prompt and 8-prompt commands, metric interpretation, and current NVFP4/mixed-FP8 results. |
| 2026-05-25 | [GLM-5.1 v4 mixed FP8 native W4A16 recipe](../glm5.1_v4.md) | Current GLM-5.1 mixed FP8PBWO L42-62 serving profile with native ModelOpt W4A16 decode, exact DockerHub tag, source branches/commits, DCP4+MTP launch recipe, KV-cache state, and KLD smoke result. |
| 2026-05-16 | [GLM-5.1 NVFP4 KLD sensitivity resume](glm51-nvfp4-kld-sensitivity-2026-05-16.md) | Complete KLD checkpoint for GLM-5.1 NVFP4/W4A16, BF16 oracle sweeps, local quant-toolkit export fixes, mixed FP8_PB_WO expert-layer checkpoint construction, public HF upload workflow, and current quality/memory tradeoff candidates. |
| 2026-05-15 | [B12X W4A16 regression checkpoint](b12x-w4a16-regression-checkpoint-2026-05-15.md) | Archive record for the experimental Luke B12X W4A16 merge, vLLM-side fixes, tested Docker image, observed GLM regression, recovery branches, and local bundle/patch backup. |
| 2026-05-15 | [GLM-5.1 v2 canonical vLLM recipe](../glm5.1_v2.md) | Canonical GLM-5.1 recipe for the rebased GLM/Kimi vLLM stack, including DCP1/2/4/8 MTP on/off decode matrices and raw artifacts under [benchmarks/glm51-v2-full-68b3569f-20260514](benchmarks/glm51-v2-full-68b3569f-20260514/). |
| 2026-05-07 | [vLLM communicator experiments for GLM-5.1 and Kimi-K2.6](vllm-communicators-kimi-glm-2026-05-07.md) | Standalone communicator log covering patched NCCL PR #2127, no-XML launch, vLLM C++ PCIe allreduce selector, b12x oneshot, RTX6K fused allreduce/add/RMS prototypes, DCP communication results, Kimi/GLM measurements, and final safe/unsafe recommendations. |
| 2026-05-04 | [GLM-5.1 vLLM DCP with NCCL PR #2127 no-XML topology fix](glm51-vllm-dcp-nccl2127-noxml-2026-05-04.md) | vLLM GLM-5.1 DCP1/2/4/8 measurements on the patched NCCL PR #2127 Docker image, exact launch command for each DCP/PCIe-allreduce variant, and comparison against official NCCL plus XML for DCP8. |
| 2026-05-03 | [GLM-5.1 cc32 SGLang vs vLLM runtime state](glm51-sglang-vllm-cc32-state-2026-05-03.md) | Current DCP=1 / TP=8 launch state for SGLang and vLLM at 32 concurrent decode requests, DockerHub image tags, exact launch overrides, and measured default/safe MTP throughput. |
| 2026-05-02 | [vLLM b12x NSA/MTP port: fast prefill, PCIe barriers, and upstream delta](vllm-b12x-nsa-mtp-port-2026-05-02.md) | GLM-5.1 NVFP4-MTP on vLLM with b12x sparse NSA, ModelOpt FP4, MTP, FP8 KV cache, PCIe allreduce, JIT/prefill fixes, and differences from vanilla b12x / Luke's SGLang patches. |
| 2026-04-20 | [Dense MLA vs NSA vs vLLM benchmark](compare-dense-mla-vs-nsa-benchmark-2026-04-20/README.md) | Quality and throughput comparison of SGLang dense MLA, SGLang NSA, and a vLLM reference path. |
