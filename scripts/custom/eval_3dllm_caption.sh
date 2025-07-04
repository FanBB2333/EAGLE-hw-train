#!/bin/bash
export RUN_NAME=`date +"%m%d"`

MODEL_PATH=./qbs/Eagle_LanguageBind/checkpoints/3D/finetune/pr_llm/finetune-3d-pointllmenc-llama3.2-3b

OUTPUT_DIR=./output/eval/${RUN_NAME}
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/3dllm_caption.json

echo ${MODEL_PATH} >> ${OUTPUT_DIR}/3dllm_caption.txt


CUDA_VISIBLE_DEVICES='0' python eval/eval_3d_conv.py \
    --model_path ${MODEL_PATH} \
    --output_path ${OUTPUT_PATH} \
    --ann_path ./qbs/Eagle_LanguageBind/dataset/3D/3D-LLM/data_part2_scene_v3/caption_val_v3_1000_convert.json \

python eval/score/3d_caption_eval.py \
    --result_file ${OUTPUT_PATH}