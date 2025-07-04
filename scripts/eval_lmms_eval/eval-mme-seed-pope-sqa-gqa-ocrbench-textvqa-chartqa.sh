export HF_DATASETS_CACHE='/home/qinbosheng/HDD/HDD1/Code/Image/EAGLE/model/hf'
# accelerate launch --num_processes=8\
#            evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks  mme,seed_bench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench_mathvista_seedbench \
#            --output_path ./logs/ 

CUDA_VISIBLE_DEVICES='0' python evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=checkpoints/Images/finetune-eagle-x1-llama3.2-1b-image_L,conv_template=llama3 \
           --tasks  mme,seed_bench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix finetune-eagle-x1-llama3.2-1b-image_L_mmbench_mathvista_seedbench \
           --output_path ./logs/ 
