#!/bin/bash
NAME=Incremental/Audio/finetune-video-qwen2.5-1.5b/finetune/pr_llm/finetune-audio-qwen2.5-1.5b

export WANDB_DISABLED="true"
export WANDB_PROJECT="eagle"
export WANDB_RUN_ID=${NAME}
export WANDB_RESUME="allow"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_audio_load_video.py ./checkpoints/$NAME

# echo "MASTER_ADDR=$MASTER_ADDR"
# n_node=$SLURM_JOB_NUM_NODES
# echo "number of nodes:" $n_node
# echo "node rank:" $SLURM_PROCID
# TORCH_DISTRIBUTED_DEBUG='DETAIL' CUDA_VISIBLE_DEVICES='0' 
CUDA_VISIBLE_DEVICES='4' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25034 \
    train_audio_load_video.py \
    --model_name_or_path ./model/LLM/Qwen2.5-1.5B-Instruct \
    --version qwen_2 \
    --data_path ./dataset/Audio/OneLLM/finetune.json \
    --image_folder ./dataset/Audio \
    --vision_tower ./model/Vision_Encoder/LanguageBind/LanguageBind_Audio_FT \
    --pretrain_mm_mlp_adapter ./checkpoints/Incremental/Audio/finetune-video-qwen2.5-1.5b/pretrain/pretrain-audio-qwen2.5-1.5b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
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
    --run_name ${NAME}
