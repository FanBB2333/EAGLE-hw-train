#!/bin/bash
export PYTHONPATH=$(pwd):/home6/fzy/repos/EAGLE
TASK=activitynet
# choices=["activitynet", "breakfast", "charades", "qvhighlights", "valor", "youcook2"]
# MODEL_PATH=./checkpoints/video_finetune_1epoch
MODEL_PATH=./checkpoints/llama_3.2b/Video/en_pr/finetune-video-llama3.2-3b
# MODEL_PATH=./checkpoints/llama_3.2b/Video/pr/finetune-video-llama3.2-1b-ori-token

echo $MODEL_PATH
echo $TASK
CUDA_VISIBLE_DEVICES='1' python eval/eval_video.py \
    --model_path ${MODEL_PATH} \
    --task ${TASK} 
