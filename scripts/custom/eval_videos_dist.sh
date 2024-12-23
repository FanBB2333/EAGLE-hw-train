#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
TASK=charades
# choices=["activitynet", "breakfast", "charades", "qvhighlights"],
MODEL_PATH=./checkpoints/video_finetune_1epoch

echo $MODEL_PATH
echo $TASK
CUDA_VISIBLE_DEVICES='1,2,3,4,5,6' accelerate launch eval/eval_video.py \
    --model_path ${MODEL_PATH} \
    --task ${TASK} \
    --distributed
