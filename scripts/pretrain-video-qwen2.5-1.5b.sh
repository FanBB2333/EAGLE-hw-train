#!/bin/bash
NAME=Baseline/Video/pretrain/pretrain-video-qwen2.5-1.5b
export WANDB_DISABLED="true"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_video1.py ./checkpoints/$NAME

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='2' python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 25033 \
    train_video1.py \
    --model_name_or_path ./model/LLM/Qwen2.5-1.5B-Instruct \
    --version plain \
    --data_path ./dataset/Video/videollava_pt/valley_llavaimage_opencv_image_ori_token.json \
    --image_folder ./dataset/Video/videollava_pt \
    --vision_tower ./model/Vision_Encoder/LanguageBind/LanguageBind_Video_FT \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --fp16 False \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
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
