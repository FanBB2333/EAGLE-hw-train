#!/bin/bash
export RUN_NAME=`date +"%m%d"`

# MODEL_PATH=./checkpoints/final_result1/audio_finetune_1epoch
# MODEL_PATH=./checkpoints/Baseline/Audio/finetune/en_pr/finetune-audio-llama3.2-3b
MODEL_PATH=./checkpoints/Audio/finetune-audio-qwen2audioenc-llama3.2-3b-onellm_qa_0611\(final\)

OUTPUT_DIR=./output/eval/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/clothocaps_.json

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/clothocaps.txt


CUDA_VISIBLE_DEVICES='0' python eval/eval_clothocaption.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}
