#!/bin/bash
NAME=Baseline/Images/finetune/pr/finetune-eagle-x1-llama3.2-3b-image_L

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_image.py ./checkpoints/$NAME

CUDA_VISIBLE_DEVICES='3' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25036 \
    train_image.py \
    --model_name_or_path ./model/LLM/Llama-3.2-3B-Instruct \
    --version llama3 \
    --data_path ./dataset/Images/Eagle-1.8M/eagle-1-sft-1_8M.json \
    --image_folder ./dataset/Images/Eagle-1.8M \
    --vision_tower ./model/Vision_Encoder/CLIP/CLIP-ViT-L-14-DataComp.XL-s13B-b90K  \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/Baseline/Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000000000000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${NAME} 