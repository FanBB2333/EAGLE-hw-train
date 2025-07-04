#!/bin/bash
NAME=3D/pretrain/pretrain-3d-pointllmenc-llama3.2-3b
export WANDB_DISABLED="true"

mkdir -p ./checkpoints/$NAME  # 创建目标文件夹（如果不存在）
cp -r ./eagle ./checkpoints/$NAME
cp -r "$0" ./checkpoints/$NAME  # 复制当前bash文件到目标目录
cp -r  train_3d.py ./checkpoints/$NAME

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='0,1,6,7' python -m torch.distributed.run \
    --nproc_per_node 4 --master_port 23711 \
    train_3d.py \
    --model_name_or_path ./model/LLM/Llama-3.2-3B-Instruct \
    --version plain \
    --data_path ./dataset/3D/PointLLM/PointLLM_brief_description_660K_filtered_convert.json \
    --image_folder ./dataset/3D \
    --vision_tower ./model/Vision_Encoder/PointBert_v1.2 \
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
    --gradient_accumulation_steps 4 \
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
