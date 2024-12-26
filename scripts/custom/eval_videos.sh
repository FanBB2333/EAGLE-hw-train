#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
TASK=breakfast
# choices=["activitynet", "breakfast", "charades", "qvhighlights"],
MODEL_PATH=./checkpoints/video_finetune_1epoch

echo $MODEL_PATH
echo $TASK
CUDA_VISIBLE_DEVICES='0' python eval/eval_video.py \
    --model_path ${MODEL_PATH} \
    --task ${TASK} 
