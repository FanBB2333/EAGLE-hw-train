#!/bin/bash
NAME=$1

export WANDB_DISABLED="true"
# export WANDB_PROJECT="eagle"
# export WANDB_RUN_ID=${NAME}
# export WANDB_RESUME="allow"

echo "MASTER_ADDR=$MASTER_ADDR"
n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID
export PATH_TO_SFT_DATA=/home1/hxl/disk/EAGLE/dataset/Eagle-1.8M
export PATH_TO_PRETRAINED_PROJECTOR=/home1/hxl/disk/EAGLE/checkpoints/1015/

# python -m torch.distributed.run \
#     --nproc_per_node 1 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    # --master_addr $MASTER_ADDR --master_port 25031 \
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 25032 \
    train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path ./model/LLM/Meta-Llama-3-8B-Instruct \
    --version llama3 \
    --data_path $PATH_TO_SFT_DATA/eagle-1-sft-1_8M.json \
    --image_folder $PATH_TO_SFT_DATA \
    --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter $PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/$NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --run_name ${NAME}  
    # --report_to wandb \
# python -m torch.distributed.run \
#     --nproc_per_node 1 \
#     --master_port 25032 \
#     train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --version llama3 \
#     --data_path $PATH_TO_SFT_DATA/eagle-sft-v1-1_8m.json \
#     --image_folder $PATH_TO_SFT_DATA/images \
#     --vision_tower "clip-448;convnext-1024;sam-1024;det-1024;pix2struct-1024" \
#     --mm_projector_type mlp2x_gelu \
#     --pretrain_mm_mlp_adapter $PATH_TO_PRETRAINED_PROJECTOR/mm_projector.bin \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/$NAME \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --run_name ${NAME}  