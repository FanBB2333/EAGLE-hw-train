#!/bin/bash
export RUN_NAME=`date +"%m%d"`

MODEL_PATH=./checkpoints/final_result1/3d_finetune_1epoch

OUTPUT_DIR=./output/eval/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/3dllm_.json

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/3dllm.txt


CUDA_VISIBLE_DEVICES='0' python eval/eval_3dllm.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH}
