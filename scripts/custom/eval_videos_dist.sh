#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
export NCCL_BLOCKING_WAIT=1
TASK=charades
# choices=["activitynet", "breakfast", "charades", "qvhighlights"],
# MODEL_PATH=./checkpoints/video_finetune_1epoch
MODEL_PATH=./checkpoints/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old
echo $MODEL_PATH
echo $TASK
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6' accelerate launch eval/eval_video.py \
    --model_path ${MODEL_PATH} \
    --task ${TASK} \
    --distributed
