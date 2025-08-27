#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2     # 单卡示例；多卡改成 0,1,2,3 等
export VLLM_USE_MODELSCOPE=False   # 默认从 HuggingFace Hub；若用 ModelScope 改成 True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve ~/models/Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 58072 \
  --max-model-len 3072 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --enable-chunked-prefill \
  --swap-space 8 \
  --served-model-name Qwen2.5-VL-72B-Instruct-AWQ





  # 可选：如需更高并发可加 --max-num-seqs 128