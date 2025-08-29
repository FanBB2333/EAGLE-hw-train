# run scripts
bash ./scripts/fzy/finetune-eagle-x1-llama3.2-3b-image-fzy-qwen2vl-llava-eagle-ocrb-nemo-en.sh
bash ./scripts/fzy/finetune-eagle-x1-llama3.2-3b-image-fzy-qwen2vl-llava-eagle-ocrb-nemo-en-pr.sh

# modify config
python config_modify.py ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en-pr
python config_modify.py ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en

# run eval image
python eval_image/eval_image_all.py --model_path ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en-pr --gpus 0 --datasets ocrbenchv2 2>&1 | tee eval_nemo_en_pr.log
python eval_image/eval_image_all.py --model_path ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en    --gpus 0 --datasets ocrbenchv2 2>&1 | tee eval_nemo_en.log

# run only_eval
# python eval_ocrbenchv2.py --model_path ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en-pr --only_eval
# python eval_ocrbenchv2.py --model_path ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en    --only_eval


# scp to 9008
scp -r -P 9008 ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en-pr fzy@900x.fanbb.top:/home7/fzy/checkpoints
scp -r -P 9008 ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en fzy@900x.fanbb.top:/home7/fzy/checkpoints

