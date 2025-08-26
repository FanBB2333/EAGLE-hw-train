#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0     # 单卡示例；多卡改成 0,1,2,3 等
export VLLM_USE_MODELSCOPE=False   # 默认从 HuggingFace Hub；若用 ModelScope 改成 True

vllm serve /home/l1ght/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 58000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --served-model-name Qwen2.5-VL-7B-Instruct
  # 可选：如需更高并发可加 --max-num-seqs 128