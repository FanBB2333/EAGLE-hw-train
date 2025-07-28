#!/bin/bash

# MVBench Evaluation Script
# This script runs MVBench evaluation with different configurations

# Set default paths
MODEL_PATH="./checkpoints/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old"
VIDEO_DIR="./dataset/MVBench/video"
OUTPUT_DIR="./output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting MVBench evaluation..."

# Option 1: Evaluate all subsets (recommended)
echo "Evaluating all MVBench subsets..."
python eval_video_qwen_mvbench.py \
    --model_path $MODEL_PATH \
    --video_dir $VIDEO_DIR \
    --output_path $OUTPUT_DIR/mvbench_all_results.json \
    --conv_template llama3 \
    --device cuda

# Option 2: Evaluate specific subset (uncomment to use)
# echo "Evaluating action_sequence subset..."
# python eval_video_qwen_mvbench.py \
#     --model_path $MODEL_PATH \
#     --subset action_sequence \
#     --video_dir $VIDEO_DIR \
#     --output_path $OUTPUT_DIR/mvbench_action_sequence_results.json \
#     --conv_template llama3 \
#     --device cuda

# Option 3: Distributed evaluation (uncomment to use)
# echo "Running distributed evaluation..."
# python eval_video_qwen_mvbench.py \
#     --model_path $MODEL_PATH \
#     --video_dir $VIDEO_DIR \
#     --output_path $OUTPUT_DIR/mvbench_all_dist_results.json \
#     --conv_template llama3 \
#     --device cuda \
#     --distributed

echo "MVBench evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check the summary file for detailed accuracy metrics by subset."
