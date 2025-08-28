bash ./scripts/fzy/finetune-eagle-x1-llama3.2-3b-image-fzy-qwen2vl-llava-eagle-ocrb-nemo-en.sh
bash ./scripts/fzy/finetune-eagle-x1-llama3.2-3b-image-fzy-qwen2vl-llava-eagle-ocrb-nemo-en-pr.sh
scp -r -P 9008 ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en-pr fzy@900x.fanbb.top:/home7/fzy/checkpoints
scp -r -P 9008 ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-nemotron-8b-en fzy@900x.fanbb.top:/home7/fzy/checkpoints