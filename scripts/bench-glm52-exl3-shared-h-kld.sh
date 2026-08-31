#!/usr/bin/env bash
set -euo pipefail

mode="${MODE:-checkpoint}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28}"
model_root="${MODEL_ROOT:-/root/models}"
model="${MODEL:-/root/models/GLM-5.2-EXL3-TR3-3.42bpw}"
reference_root="${REFERENCE_ROOT:-/root/kld/glm52_current_bf16_returnlogits_ref_20260708T000424Z/ref}"
output_root="${OUTPUT_ROOT:-/root/kld/glm52-exl3-shared-h-r28}"
gpus="${GPUS:-0,1,2,3}"
docker_gpus="\"device=${gpus}\""
repeats="${REPEATS:-3}"
cache_root="${CACHE_ROOT:-/root/cache/glm52-exl3-shared-h-kld}"
tokens_file="${TOKENS_FILE:-${repo_root}/benchmarks/data/glm52-kld-tokens-2048.json}"

case "${mode}" in
  checkpoint)
    label="checkpoint-only-matched-fp8-kv"
    kv_cache_dtype=fp8
    quantization_config=null
    extra_env=()
    ;;
  runtime-fp8)
    label="runtime-k6-matched-fp8-kv"
    kv_cache_dtype=fp8
    quantization_config='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
    extra_env=(-e VLLM_EXL3_ONLINE_TRELLIS_BITS=6)
    ;;
  runtime)
    label="runtime-k6-nvfp4-kv"
    kv_cache_dtype=nvfp4_ds_mla
    quantization_config='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*\\.fused_qkv_a_proj$","re:.*\\.q_a_proj$","re:.*kv_a_proj_with_mqa","re:.*\\.mlp\\.gate$","model.layers.78.eh_proj","lm_head"]}'
    extra_env=(-e VLLM_EXL3_ONLINE_TRELLIS_BITS=6)
    ;;
  *)
    printf 'MODE must be checkpoint, runtime-fp8, or runtime; got %s\n' "${mode}" >&2
    exit 2
    ;;
esac

output_dir="${output_root}/${label}"
mkdir -p "${output_dir}" "${cache_root}"
docker rm -f glm52-exl3-shared-h-kld >/dev/null 2>&1 || true

docker run --rm \
  --name glm52-exl3-shared-h-kld \
  --gpus "${docker_gpus}" \
  --network host \
  --ipc host \
  --init \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --ulimit nofile=1048576:1048576 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1 \
  -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  "${extra_env[@]}" \
  -v "${model_root}:${model_root}:ro" \
  -v "${HF_CACHE:-/root/.cache/huggingface}:/root/.cache/huggingface" \
  -v "${reference_root}:/reference:ro" \
  -v "${tokens_file}:/tokens-2048.json:ro" \
  -v "${repo_root}/scripts/glm52_exl3_shared_h_kld.py:/runner.py:ro" \
  -v "${output_dir}:/results" \
  -v "${cache_root}:/cache" \
  --entrypoint /opt/venv/bin/python \
  "${image}" /runner.py \
    --label "${label}" \
    --model "${model}" \
    --reference-logits /reference \
    --tokens-file /tokens-2048.json \
    --output-dir /results \
    --quantization-config "${quantization_config}" \
    --tensor-parallel-size 4 \
    --kv-cache-dtype "${kv_cache_dtype}" \
    --repeats "${repeats}"
