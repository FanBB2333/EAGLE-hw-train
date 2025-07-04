#!/bin/bash
export RUN_NAME=`date +"%m%d"`

MODEL_PATH=./checkpoints/3D/finetune/pr_llm/finetune-3d-pointllmenc-llama3.2-3b

OUTPUT_DIR=./output/eval/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/scanqa.json

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/scanqa.txt


CUDA_VISIBLE_DEVICES='0' python eval/eval_3d_conv.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --ann_path ./dataset/3D/ScanQA/ScanQA_v1.0_val_convert.json \

python eval/score/3d_caption_eval.py \
    --result_file ${OUTPUT_PATH}