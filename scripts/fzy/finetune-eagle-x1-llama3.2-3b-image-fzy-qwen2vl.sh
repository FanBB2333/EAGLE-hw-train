#!/bin/bash
# NAME=Baseline/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-docvqa
NAME=disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_image.py ./checkpoints/$NAME

# echo "MASTER_ADDR=$MASTER_ADDR"
# n_node=$SLURM_JOB_NUM_NODES
# echo "number of nodes:" $n_node
# echo "node rank:" $SLURM_PROCID
# TORCH_DISTRIBUTED_DEBUG='DETAIL' CUDA_VISIBLE_DEVICES='0' 
# /home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/
CUDA_VISIBLE_DEVICES='2,3' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25030 \
    train_image.py \
    --model_name_or_path ./model/LLM/Llama-3.2-3B-Instruct \
    --version llama3 \
    --data_path ./dataset/Images/combined/combined.json \
    --image_folder ./dataset/Images/combined \
    --vision_tower ./model/disk2/Qwen/Qwen2-VL-2B-Instruct  \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --num_train_epochs 1 \
    --run_name ${NAME} \


    # --evaluation_strategy "no" \
    # --pretrain_mm_mlp_adapter ./checkpoints/Baseline/Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L/mm_projector.bin \
