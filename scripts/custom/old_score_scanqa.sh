#!/bin/bash


OUTPUT_DIR=./output/eval/onellm
mkdir -p ${OUTPUT_DIR}
OUTPUT_PATH=${OUTPUT_DIR}/scanqa_score.txt

INPUT_PATH=./output/eval/onellm/eval_scanqa.json


CUDA_VISIBLE_DEVICES='0' python eval/score/scanqa.py \
    --json_path ${INPUT_PATH} > ${OUTPUT_PATH}