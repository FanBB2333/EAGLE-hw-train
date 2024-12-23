#!/bin/bash
TASKS=pointllm
# MODEL_PATH=./checkpoints/final_result1/3d_finetune_1epoch
MODEL_PATH=./checkpoints/video_finetune_1epoch
echo $MODEL_PATH
echo $TASKS
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
CUDA_VISIBLE_DEVICES='0' python eval/eval_3d.py \
    --model_path ${MODEL_PATH} 
echo $MODEL_PATH
echo $TASKS