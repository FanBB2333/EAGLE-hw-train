#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3      # 单卡示例；多卡改成 0,1,2,3 等
export VLLM_USE_MODELSCOPE=False   # 默认从 HuggingFace Hub；若用 ModelScope 改成 True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve /home6/fzy/models/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 58007 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --enable-chunked-prefill \
  --swap-space 8 \
  --served-model-name Qwen2.5-VL-7B-Instruct
  # 可选：如需更高并发可加 --max-num-seqs 128