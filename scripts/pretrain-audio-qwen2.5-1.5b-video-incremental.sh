#!/bin/bash
NAME=Incremental/Audio/pretrain-audio-qwen2.5-1.5b/pretrain/pretrain-audio-qwen2.5-1.5b
export WANDB_DISABLED="true"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_audio_load_video.py ./checkpoints/$NAME

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='3,4' python -m torch.distributed.run \
    --nproc_per_node 2 --master_port 25033 \
    train_audio_load_video.py \
    --model_name_or_path ./model/LLM/Qwen2.5-1.5B-Instruct \
    --version plain \
    --data_path ./dataset/Audio/WavCaps/pretrain_filtered.json \
    --image_folder ./dataset/Audio/WavCaps \
    --vision_tower ./model/Vision_Encoder/LanguageBind/LanguageBind_Audio_FT \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 1e-3 \
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
    --run_name ${NAME}
