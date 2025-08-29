import os
import json
import argparse

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

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description='批量修改config.json文件中的image_aspect_ratio参数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python config_modify.py ./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-qwen2.5_VL-72b-en-pr
        '''
    )
    
    parser.add_argument(
        '--folder_path', '-f',
        default='./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-ocrb-qwen2.5_VL-72b-en-pr',
        help='要处理的文件夹路径'
    )
    
    args = parser.parse_args()
    
    folder_to_check = args.folder_path
    
    if not os.path.exists(folder_to_check):
        parser.error(f"文件夹 '{folder_to_check}' 不存在")
    
    print(f"开始处理文件夹: {folder_to_check}")
    update_config_files(folder_to_check)
    print("处理完成")

if __name__ == "__main__":
    main()