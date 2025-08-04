#!/usr/bin/env python3
"""
使用示例：eval_video_all.py 支持 eval_video_qwen 数据集

展示如何使用修改后的 eval_video_all.py 来运行 eval_video_qwen 中的多个数据集
"""

# 使用示例命令：

# 1. 运行单个数据集
# python eval_video_all.py --datasets charades

# 2. 运行多个 eval_video_qwen 数据集
# python eval_video_all.py --datasets charades,mvbench,activitynet

# 3. 运行混合数据集（包括 acqa 和 eval_video_qwen 数据集）
# python eval_video_all.py --datasets acqa,charades,mvbench

# 4. 运行所有支持的数据集
# python eval_video_all.py --datasets all

# 5. 指定自定义模型路径
# python eval_video_all.py --datasets charades --model_path /path/to/your/model

# 6. 指定输出目录
# python eval_video_all.py --datasets mvbench --output_dir ./my_results

# 支持的数据集列表：
SUPPORTED_DATASETS = {
    # 来自 eval_acqa.py
    "acqa": "ActivityNetQA - Video question answering",
    
    # 来自 eval_video_qwen.py
    "activitynet": "ActivityNet Captions - Video captioning and temporal localization",
    "breakfast": "Breakfast Actions - Step-by-step action recognition", 
    "charades": "Charades Actions - Action localization and description",
    "qvhighlights": "QV Highlights - Query-based video highlight detection",
    "valor": "VALOR32K - Video and language understanding",
    "youcook2": "YouCook2 - Instructional video understanding",
    "mvbench": "MVBench - Multi-view video understanding benchmark",
}

print("支持的数据集：")
for dataset, description in SUPPORTED_DATASETS.items():
    print(f"  - {dataset}: {description}")
