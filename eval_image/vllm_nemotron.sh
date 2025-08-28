#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0     # 单卡示例；多卡改成 0,1,2,3 等
export VLLM_USE_MODELSCOPE=False   # 默认从 HuggingFace Hub；若用 ModelScope 改成 True

vllm serve ~/models/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1 \
  --host 0.0.0.0 \
  --port 58008 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --served-model-name Llama-3.1-Nemotron-Nano-VL-8B-V1 \
  --trust_remote_code \
  # 允许仓库中的自定义代码运行（修复: 请传入 trust_remote_code=True），
  # 可选：如需更高并发可加 --max-num-seqs 128