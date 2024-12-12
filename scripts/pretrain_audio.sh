#!/bin/bash
NAME=audio_try

export WANDB_DISABLED="true"

CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25033 \
    train_audio.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path ./model/LLM/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path ./dataset/AudioSetCaps/example.csv \
    --audio_folder ./dataset/AudioSetCaps/example \
    --audio_tower "./model/LanguageBind_Audio_FT" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_audio_select_layer -2 \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
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
    --run_name ${NAME}
