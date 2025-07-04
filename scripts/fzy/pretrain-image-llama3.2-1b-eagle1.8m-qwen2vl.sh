#!/bin/bash
# NAME=Baseline/Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L_fzy-1.8m-qwen2vl-batch
NAME=Images/pretrain/pretrain-eagle-x1-llama3.2-3b_image_L_fzy-llava-qwen2vl

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_image.py ./checkpoints/$NAME

CUDA_VISIBLE_DEVICES='3,5,6,7' python -m torch.distributed.run \
    --nproc_per_node 4 --master_port 25035 \
    train_image.py \
    --model_name_or_path ./model/LLM/Llama-3.2-3B-Instruct \
    --version plain \
    --data_path ./dataset/Images/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./dataset/Images/llava_pretrain/images \
    --vision_tower ./model/disk2/Qwen/Qwen2-VL-2B-Instruct \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --save_strategy "epoch" \
    --learning_rate 1e-5 \
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


    # --evaluation_strategy "no" \
