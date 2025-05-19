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
folder_to_check = '/home6/fzy/repos/EAGLE/checkpoints/llama_3.2b/Video'
update_config_files(folder_to_check)