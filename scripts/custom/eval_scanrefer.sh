#!/bin/bash
export RUN_NAME=`date +"%m%d"`

MODEL_PATH=./checkpoints/3D/finetune/pr_llm/finetune-3d-pointllmenc-llama3.2-3b

OUTPUT_DIR=./output/eval/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/scanrefer.json

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/scanrefer.txt


CUDA_VISIBLE_DEVICES='1' python eval/eval_3d_conv.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --ann_path ./dataset/3D/ScanRefer/ScanRefer_filtered_val_convert.json \

python eval/score/3d_caption_eval.py \
    --result_file ${OUTPUT_PATH}