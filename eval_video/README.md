# Video Evaluation Framework

这是一个用于运行多种视频评估任务的综合框架，参照 `eval_image_all.py` 的结构设计。

## 功能特性

- 🚀 自动结果保存，包含时间戳和元数据
- 🔄 支持顺序和并行执行
- 📊 详细的结果格式化和统计摘要
- 🔌 可扩展的框架，便于添加新的视频评估数据集
- 📁 组织化的结果存储

## 当前支持的数据集

- **ACQA (ActivityNetQA)**: 基于ActivityNet的视频问答任务

## 使用方法

### 基本用法

```bash
# 运行 ActivityNetQA 评估
python eval_video_all.py --datasets acqa

# 使用自定义模型路径
python eval_video_all.py --datasets acqa --model_path /path/to/your/model

# 使用自定义输出目录
python eval_video_all.py --datasets acqa --output_dir ./my_results

# 指定使用的GPU
python eval_video_all.py --datasets acqa --gpus "0,1"
```

### 高级用法

```bash
# 运行所有可用的评估（目前只有ACQA）
python eval_video_all.py --datasets all

# 将来支持多个数据集时的用法示例
python eval_video_all.py --datasets acqa,mvbench

# 使用顺序执行模式（默认）
python eval_video_all.py --datasets acqa --sequential
```

## 参数说明

- `--model_path`: 预训练模型的路径 (默认: `./checkpoints/Videos/finetune-video-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle`)
- `--datasets`: 要评估的数据集，可选: `all`, `acqa` 或逗号分隔的列表
- `--gpus`: 使用的GPU ID，逗号分隔 (默认: `"0"`)
- `--output_dir`: 自定义输出目录 (可选)
- `--sequential`: 顺序执行而非并行执行 (默认: True)

## 结果文件

评估结果将保存为 JSON 文件，包含：

- **元数据**: 时间戳、模型信息、评估配置
- **统计摘要**: 关键指标的汇总
- **详细结果**: 完整的评估输出

结果文件默认保存在 `eval_video/res_folder/videos/` 目录下。

## 框架结构

```
eval_video/
├── eval_video_all.py      # 主框架文件
├── eval_acqa.py           # ActivityNetQA 评估脚本
├── test_framework.py      # 框架测试脚本
├── res_folder/
│   └── videos/            # 评估结果存储目录
└── README.md              # 本文件
```

## 添加新的数据集评估

要添加新的数据集评估，请按以下步骤：

1. **创建评估脚本** (例如 `eval_newdataset.py`):
   ```python
   def evaluate_with_results(model_path, datasets=None):
       # 实现评估逻辑
       # 返回格式：
       return {
           'status': 'completed',
           'metric1': value1,
           'metric2': value2,
           'output_file': output_file_path
       }
   ```

2. **更新 `eval_video_all.py`**:
   ```python
   # 在 dataset_scripts 字典中添加新数据集
   dataset_scripts = {
       "acqa": "eval_acqa.py",
       "newdataset": "eval_newdataset.py",  # 添加这一行
   }
   
   # 在 run_evaluation_internal 函数中添加导入
   elif script_name == "eval_newdataset.py":
       from eval_newdataset import evaluate_with_results
   ```

## 测试框架

运行测试脚本验证框架是否正常工作：

```bash
python eval_video/test_framework.py
```

## 依赖要求

- PyTorch
- transformers
- datasets
- tqdm
- PIL
- 其他 EAGLE 项目依赖

## 注意事项

1. 确保 CUDA_VISIBLE_DEVICES 环境变量正确设置
2. 视频文件路径需要正确配置
3. 模型路径必须指向有效的预训练模型
4. 确保有足够的存储空间保存结果文件

## 扩展计划

- [ ] 添加 MVBench 评估支持
- [ ] 添加 YouCook2 评估支持
- [ ] 添加 Charades 评估支持
- [ ] 支持自定义评估指标
- [ ] 添加结果可视化功能
