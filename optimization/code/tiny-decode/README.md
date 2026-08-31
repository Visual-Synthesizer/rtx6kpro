# tiny-decode standalone kit

- `tiny_moe.py` — the final 2-kernel Triton implementation (matches vLLM
  `b12x_tiny_decode.py` commit `8e6e417c` modulo the integration wrapper).
- `tiny_moe_test.py` — correctness (fp32 oracle + dynamic-kernel cross-check)
  and graph-replay timing driver. Runs against the b12x benchmarks harness.
- `tiny_moe_fused_v4_slow.py` — the abandoned single-kernel fused variant
  (correct but 96-111 us; kept for reference of what did NOT work).

Run (GPU 6, in-container):
```bash
docker run --rm --gpus all --runtime nvidia --ipc host \
  -e CUDA_VISIBLE_DEVICES=6 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -v <b12x-checkout>/benchmarks:/bench:ro -v $(pwd):/work:ro --entrypoint /bin/sh \
  voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x77bd50e-overlay-cu132-20260702 \
  -lc 'python3 /work/tiny_moe_test.py'
```
See ../../b12x-w4a8mx-tiny-decode-kernel.md for the full writeup.
