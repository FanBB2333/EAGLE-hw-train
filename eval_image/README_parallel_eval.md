# 并行模型评估脚本使用说明

## 概述

这个脚本系统用于并行评估多个合并模型的性能。主要包含两个脚本：

1. `eval_image_all.sh` - 主评估脚本
2. `monitor_eval.sh` - 监控和结果查看脚本

## 文件结构

```
eval_image/
├── eval_image_all.sh       # 主并行评估脚本
├── eval_image_all.py       # Python评估脚本
├── monitor_eval.sh         # 监控脚本
├── logs/                   # 日志输出目录
│   ├── eval_*.log         # 评估日志
│   └── eval_*.err         # 错误日志
└── res_folder/            # 结果输出目录
    └── images/
        └── [model_name]/
            └── summary_*.json
```

## 使用方法

### 1. 启动并行评估

```bash
# 评估所有数据集
./eval_image_all.sh

# 评估特定数据集
./eval_image_all.sh mmlu
./eval_image_all.sh "mmlu,mme"

# 查看帮助
./eval_image_all.sh --help
```

### 2. 监控评估进度

```bash
# 查看当前状态
./monitor_eval.sh

# 查看错误日志
./monitor_eval.sh errors

# 查看结果摘要
./monitor_eval.sh results

# 实时跟踪特定模型的日志
./monitor_eval.sh tail 0.9_0.1

# 清理旧日志文件
./monitor_eval.sh clean
```

## 配置说明

### 模型配置

脚本会自动评估以下模型（位于 `/home6/fzy/repos/EAGLE/checkpoints/Images/merged_model/renamed/`）：

- 0.6_0.4
- 0.7_0.3
- 0.8_0.2
- 0.9_0.1
- 0.99_0.01
- 0.999_0.001
- 0.9999_0.0001

### GPU 分配

脚本会自动将模型分配到不同的GPU上：
- 支持GPU 0-7
- 模型按顺序循环分配到可用GPU
- 每个GPU并行运行一个评估任务

### 数据集选项

支持的数据集：
- `all` - 所有数据集（默认）
- `mmlu` - MMLU基准测试
- `mme` - MME评估
- `docvqa` - DocVQA数据集
- `textvqa` - TextVQA数据集
- `chartqa` - ChartQA数据集
- `ocrbenchv2` - OCRBench v2

## 输出文件

### 日志文件

每个模型评估会生成两个日志文件：
- `eval_{model_name}_gpu{gpu_id}_proc{process_id}.log` - 标准输出日志
- `eval_{model_name}_gpu{gpu_id}_proc{process_id}.err` - 错误日志

### 结果文件

评估结果保存在：
```
res_folder/images/{model_name}/
├── {dataset_name}/
│   └── [具体数据集结果]
└── summary_{datasets}_{timestamp}.json
```

## 示例使用流程

1. **启动评估**：
```bash
cd /home6/fzy/repos/EAGLE/eval_image
./eval_image_all.sh mmlu
```

2. **监控进度**：
```bash
# 在另一个终端中
./monitor_eval.sh
```

3. **查看特定模型的实时日志**：
```bash
./monitor_eval.sh tail 0.9_0.1
```

4. **评估完成后查看结果**：
```bash
./monitor_eval.sh results
```

## 故障排除

### 常见问题

1. **模型路径不存在**
   - 检查 `/home6/fzy/repos/EAGLE/checkpoints/Images/merged_model/renamed/` 目录
   - 确保符号链接正确

2. **GPU内存不足**
   - 调整GPU分配策略
   - 减少并行评估的模型数量

3. **Python环境问题**
   - 确保在正确的conda环境中
   - 检查依赖包是否安装

### 日志检查

```bash
# 查看错误
./monitor_eval.sh errors

# 查看特定模型的详细日志
tail -f logs/eval_0.9_0.1_gpu0_proc1.log
```

## 性能优化

- 脚本会在启动评估之间添加2秒延迟，避免系统过载
- 每个评估使用独立的GPU，避免显存冲突
- 日志文件按进程编号命名，便于追踪

## 注意事项

1. 确保有足够的磁盘空间存储日志和结果
2. 评估过程可能需要较长时间，建议在screen或tmux中运行
3. 监控系统资源使用情况，避免过载
4. 定期清理旧的日志文件以节省空间
