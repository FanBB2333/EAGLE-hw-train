# MODEL_PATH=$1
# MODEL_NAME=$2
# CONV_MODE=$3

# accelerate launch --num_processes=8\
#            evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks  mme,seed_bench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench_mathvista_seedbench \
#            --output_path ./logs/ 

# MODEL_PATH=./model/Eagle-X5-7B
# MODEL_PATH=./model/Eagle-X4-8B-Plus
# MODEL_PATH=./checkpoints/final_result/image/finetune-eagle-x1-llama3.2-1b-image_L
# MODEL_PATH=./checkpoints/final_result/video/video_finetune_1epoch
# MODEL_PATH=./checkpoints/final_result1/video_finetune_1epoch
MODEL_PATH=./checkpoints/final_result1/audio_finetune_1epoch
MODEL_NAME=eagle
# CONV_MODE=vicuna_v1
CONV_MODE=llama3
# TASKS=mme
# TASKS=mme,seedbench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa,docvqa
# TASKS=activitynetqa,egoschema,mvbench,perceptiontest_test_mc,videomme,videochatgpt,youcook2_val
# TASKS=mvbench
# TASKS=youcook2_val
# TASKS=librispeech
TASKS=clotho_aqa_test

CUDA_VISIBLE_DEVICES='1' accelerate launch --num_processes=1 \
           evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks ${TASKS} \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench \
           --output_path ./output/eval/ 
echo $MODEL_PATH
echo $TASKS
# CUDA_VISIBLE_DEVICES='1' python evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks ${TASKS} \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench \
#            --output_path ./output/eval/ 
        #    --verbosity=DEBUG
echo $MODEL_PATH
echo $TASKS