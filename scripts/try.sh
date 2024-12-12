#!/bin/bash
export OUTPUT_FOLDER=/home1/hxl/disk/EAGLE/output

export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25033 \
    train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./model/LLM/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path ./dataset/llava_pretrain/blip_laion_cc_sbu_10.json \
    --image_folder ./dataset/llava_pretrain/images \
    --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/Try \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name Try > ${OUTPUT_FOLDER}/try.txt
