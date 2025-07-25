import os
import json

def update_config_files(folder_path):
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'config.json':
                file_path = os.path.join(root, file)
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                    # 检查并更新"image_aspect_ratio"
                    if 'image_aspect_ratio' in config:
                        config['image_aspect_ratio'] = "disabled_by_hxl"
                        print(f'更新文件: {file_path}')

                        # 写入修改后的JSON文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, ensure_ascii=False, indent=2)
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f'无法处理文件 {file_path}: {e}')

# 使用示例：替换为您要检查的文件夹路径
# folder_to_check = '/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/checkpoints/Baseline/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-added-4'
# folder_to_check = '/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-added-acqa'
# folder_to_check = '/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/checkpoints/Baseline/Audio/finetune/pr_llm/finetune-audio-qwen2audioenc-llama3.2-3b-onellm_qa_0611'
# folder_to_check = '/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-added-5'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-llava'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/pretrain/pretrain-video-llama3.2-3b-fzy-qwen2vl-eagle'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-added-acqa-inc'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-900'
# folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen25vl-batch-llava-eagle'
folder_to_check = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-epoch2'
update_config_files(folder_to_check)