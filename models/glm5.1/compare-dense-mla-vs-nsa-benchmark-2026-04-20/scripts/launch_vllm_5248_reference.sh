#!/usr/bin/env bash
set -euo pipefail
python3 -m vllm.entrypoints.openai.api_server   --model lukealonso/GLM-5.1-NVFP4   --host 0.0.0.0   --port 5248   --served-model-name abtest   --trust-remote-code   --tensor-parallel-size 8   --gpu-memory-utilization 0.93   --max-num-batched-tokens 16384   --max-num-seqs 16   --kv-cache-dtype bfloat16   --enable-prefix-caching   --enable-chunked-prefill   --reasoning-parser glm45   --moe-backend b12x   --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"probabilistic","moe_backend":"triton"}'
