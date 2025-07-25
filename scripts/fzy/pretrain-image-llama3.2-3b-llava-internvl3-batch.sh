#!/bin/bash
# NAME=Baseline/Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L_fzy-llava-qwen2vl-batch
NAME=Baseline/Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L_fzy-llava-internvl3

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_image.py ./checkpoints/$NAME

CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25035 \
    train_image.py \
    --model_name_or_path ./model/LLM/Llama-3.2-3B-Instruct \
    --version plain \
    --data_path ./dataset/Images/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./dataset/Images/llava_pretrain/images \
    --vision_tower ./model/disk2/OpenGVLab/InternViT-300M-448px-V2_5 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${NAME}


    # --save_strategy "epoch" \
    # --evaluation_strategy "no" \
    # --save_strategy "steps" \
    # --save_steps 1000 \