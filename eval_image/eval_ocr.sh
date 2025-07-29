#!/bin/bash
export RUN_NAME=`date +"%m%d"`

# MODEL_PATH=/home1/hxl/disk/EAGLE/checkpoints/final_result/image/finetune-eagle-x1-llama3.2-1b-image_L
# MODEL_PATH=/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-doc-chart
MODEL_PATH=/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle/checkpoint-20000
DATA_NAME=all
OUTPUT_DIR=./eagle_ocr/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/${DATA_NAME}.json\

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/ocr.txt


CUDA_VISIBLE_DEVICES="4" python eval_eagle.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --dataset_name ${DATA_NAME} \
