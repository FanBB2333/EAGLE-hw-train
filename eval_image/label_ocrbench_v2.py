"""
OCRBench v2 数据集标注工具

该脚本用于使用多模态大语言模型对OCRBench v2数据集进行自动标注，生成训练数据格式。

主要功能：
1. 使用VLLM或transformers加载多模态模型
2. 对OCRBench v2数据集中的图片和问题进行推理
3. 生成标准的对话格式训练数据
4. 复制并组织图片文件

支持的数据文件：
- OCRBench_v2.json: 原始数据
- OCRBench_v2_new_5.json: 精选5k样本
- OCRBench_v2_new_all.json: 扩展数据集

输出格式：
- labeled_ocrbench_v2.json: 标注结果 
- images/: 复制的图片文件
- labeling_stats.json: 统计信息

使用方法：
python label_ocrbench_v2.py --model_path /path/to/model --json_data_file OCRBench_v2_new_5.json
"""

import vllm
import json
import os
import shutil
import torch
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

QWEN25VL7B = "/home6/fzy/models/Qwen2.5-VL-7B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_model_name(model_name_or_path):
    t = model_name_or_path.split("/")[-1]
    t = t.replace('-', '_')
    return t


def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    """Helper function to pad sequences (from eval_ocrbenchv2.py)"""
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def label_ocrv2(model_name_or_path: str, json_data_file: str = "OCRBench_v2.json", output_dir: str = None, force_vllm: bool = True):
    """
    使用VLLM对OCRBench v2数据集进行标注
    
    Args:
        model_name_or_path: 模型路径
        json_data_file: OCRBench v2数据文件名
        output_dir: 输出目录，如果为None则使用默认目录
        force_vllm: 是否强制使用VLLM，如果为True则在VLLM失败时不自动fallback
    """
    # 1. load vllm model using all the GPUs
    print(f"Loading VLLM model from {model_name_or_path}...")
    
    try:
        # 设置VLLM参数 - 根据模型类型调整参数
        from vllm import LLM, SamplingParams
        import torch
        
        # 获取可用GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        
        # 更保守的VLLM参数设置，减少失败概率
        vllm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": min(gpu_count, 4) if gpu_count > 1 else 1,  # 限制最大并行度
            "trust_remote_code": True,
            "max_model_len": 2048,  # 减少内存使用
            "gpu_memory_utilization": 0.8,  # 预留一些GPU内存
        }
        
        # 对于已知的多模态模型添加特定配置
        if any(name in model_name_or_path.lower() for name in ["qwen", "llava", "eagle"]):
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 1}
        
        llm = LLM(**vllm_kwargs)
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,  # 减少生成长度
            stop=None
        )
        
        print("VLLM model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading VLLM model: {e}")
        if force_vllm:
            print("force_vllm=True, stopping execution. Use --use_fallback to force transformers method.")
            raise e
        else:
            print("Falling back to transformers method...")
            return label_ocrv2_fallback(model_name_or_path, json_data_file, output_dir)
    
    # 2. do inference
    print("Loading OCRBench v2 dataset...")
    
    # 加载数据集
    json_data_path = os.path.join(str(PROJECT_ROOT / 'eval_image/OCRBench_v2'), json_data_file)
    with open(json_data_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    img_dir = str(PROJECT_ROOT / 'eval_image/OCRBench_v2')
    
    print(f"Loaded {len(json_data)} samples from {json_data_file}")
    
    # 准备输出目录
    if output_dir is None:
        model_name = get_model_name(model_name_or_path)
        current_date = datetime.now().strftime("%m%d")
        output_dir = str(PROJECT_ROOT / f'eval_image/labeled_data/{model_name}_{current_date}')
    
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # 进行推理
    results = []
    
    # 准备输入数据
    inputs = []
    valid_indices = []
    
    for i, data_dict in enumerate(json_data):
        try:
            # 构建问题
            question = data_dict['question']
            # 添加提示词以提高回答质量
            question_with_prompt = question + '\nAnswer the question using a single word or phrase.'
            
            # 检查图片是否存在
            image_path = os.path.join(img_dir, data_dict['image_path'])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # 对于VLLM，需要准备适当的输入格式
            # 这里假设使用文本提示，实际的多模态处理可能需要不同的格式
            prompt = f"<image>{question_with_prompt}"
            
            inputs.append({
                "prompt": prompt,
                "image_path": image_path,
                "data_index": i
            })
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error preparing input for sample {i}: {e}")
            continue
    
    print(f"Prepared {len(inputs)} valid inputs for inference...")
    
    # 批量推理
    try:
        # 注意：这里的实现取决于具体的VLLM版本和多模态支持
        # 对于不支持多模态的版本，可能需要fallback到transformers
        print("Starting VLLM inference...")
        prompts = [inp["prompt"] for inp in inputs]
        
        # 分批处理以避免内存问题
        batch_size = 32  # 可以根据GPU内存调整
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(batch_outputs)
        
        outputs = all_outputs
        
        # 处理输出
        for i, (inp, output) in enumerate(zip(inputs, outputs)):
            try:
                data_index = inp["data_index"]
                data_dict = json_data[data_index]
                
                if output.outputs and len(output.outputs) > 0:
                    prediction = output.outputs[0].text.strip()
                else:
                    prediction = ""
                    print(f"Warning: Empty output for sample {data_index}")
                
                # 复制图片到输出目录
                image_filename = f"{data_dict['id']}_{os.path.basename(data_dict['image_path'])}"
                image_output_path = os.path.join(images_output_dir, image_filename)
                shutil.copy2(inp["image_path"], image_output_path)
                
                # 构建结果项
                result_item = {
                    "id": str(data_dict['id']),
                    "conversations": [
                        {
                            "from": "human",
                            "value": data_dict['question'] + "\n<image>"
                        },
                        {
                            "from": "gpt", 
                            "value": prediction
                        }
                    ],
                    "image_abs": image_output_path,
                    "image": f"images/{image_filename}",
                    "original_question": data_dict['question'],
                    "original_answers": data_dict.get('answers', []),
                    "dataset_name": data_dict.get('dataset_name', ''),
                    "type": data_dict.get('type', ''),
                    "image_path": data_dict['image_path']
                }
                
                results.append(result_item)
                
            except Exception as e:
                print(f"Error processing output for sample {i}: {e}")
                continue
                
    except Exception as e:
        print(f"Error during VLLM inference: {e}")
        if force_vllm:
            print("force_vllm=True, stopping execution. Use --use_fallback to force transformers method.")
            raise e
        else:
            print("Falling back to transformers-based inference...")
            return label_ocrv2_fallback(model_name_or_path, json_data_file, output_dir)
    
    # 3. store the result into json, sample item as follows and store the images
    output_json_path = os.path.join(output_dir, 'labeled_ocrbench_v2.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Labeled {len(results)} samples")
    print(f"Results saved to: {output_json_path}")
    print(f"Images saved to: {images_output_dir}")
    
    # 保存统计信息
    stats = {
        "total_samples": len(json_data),
        "labeled_samples": len(results),
        "model_name": model_name_or_path,
        "json_data_file": json_data_file,
        "timestamp": datetime.now().isoformat()
    }
    
    stats_path = os.path.join(output_dir, 'labeling_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return results


def label_ocrv2_fallback(model_name_or_path: str, json_data_file: str = "OCRBench_v2.json", output_dir: str = None):
    """
    使用transformers作为fallback方法对OCRBench v2数据集进行标注
    """
    import sys
    from pathlib import Path
    
    # 添加项目根目录到path
    sys.path.append(str(PROJECT_ROOT))
    
    try:
        # 使用类似eval_ocrbenchv2.py的方法
        from eagle.model.builder import load_pretrained_model
        from eagle.mm_utils import process_images, tokenizer_image_token
        from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from eagle.conversation import conv_templates
        import torch
        
        print("Loading model using transformers...")
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path=model_name_or_path,
            model_base=None,
            model_name="eagle"  # 可以根据需要调整
        )
        
        model.eval()
        
        # 加载数据集
        json_data_path = os.path.join(str(PROJECT_ROOT / 'eval_image/OCRBench_v2'), json_data_file)
        with open(json_data_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        img_dir = str(PROJECT_ROOT / 'eval_image/OCRBench_v2')
        
        # 准备输出目录
        if output_dir is None:
            model_name = get_model_name(model_name_or_path)
            current_date = datetime.now().strftime("%m%d")
            output_dir = str(PROJECT_ROOT / f'eval_image/labeled_data/{model_name}_{current_date}')
        
        os.makedirs(output_dir, exist_ok=True)
        images_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_output_dir, exist_ok=True)
        
        print(f"Processing {len(json_data)} samples...")
        results = []
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for i, data_dict in enumerate(tqdm(json_data, desc="Labeling with transformers")):
            try:
                # 加载图片
                image_path = os.path.join(img_dir, data_dict['image_path'])
                if not os.path.exists(image_path):
                    continue
                
                image = Image.open(image_path).convert('RGB')
                
                # 处理图片
                image_tensor = process_images(
                    images=[image],
                    image_processor=image_processor,
                    model_cfg=model.config
                )
                image_tensor = image_tensor.to(dtype=torch.float16, device=device)
                
                # 处理image_grid_thw
                if hasattr(image_tensor, "image_grid_thw"):
                    image_grid_thw = image_tensor['image_grid_thw']
                    image_tensor = image_tensor['pixel_values']
                else:
                    image_grid_thw = None
                
                # 构建问题
                question = data_dict['question'] + '\nAnswer the question using a single word or phrase.'
                
                if DEFAULT_IMAGE_TOKEN not in question:
                    question = DEFAULT_IMAGE_TOKEN + '\n' + question
                
                # 构建对话
                conv = conv_templates["llama3"].copy()  # 可以根据模型调整
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                # Tokenize
                input_ids = tokenizer_image_token(
                    prompt_question, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    return_tensors="pt"
                )
                
                pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                input_ids = pad_sequence(
                    tokenizer=tokenizer,
                    input_ids=[input_ids], 
                    batch_first=True, 
                    padding_value=pad_token_ids
                ).to(device)
                
                attention_masks = input_ids.ne(pad_token_ids).to(device)
                
                # Generate
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=100,
                        use_cache=True,
                        modality='image',
                        image_grid_thw=image_grid_thw
                    )
                
                # Decode
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                # 提取生成的部分
                if prompt_question in output_text:
                    prediction = output_text.replace(prompt_question, "").strip()
                else:
                    prediction = output_text.strip()
                
                # 复制图片
                image_filename = f"{data_dict['id']}_{os.path.basename(data_dict['image_path'])}"
                image_output_path = os.path.join(images_output_dir, image_filename)
                shutil.copy2(image_path, image_output_path)
                
                # 构建结果项
                result_item = {
                    "id": str(data_dict['id']),
                    "conversations": [
                        {
                            "from": "human",
                            "value": data_dict['question'] + "\n<image>"
                        },
                        {
                            "from": "gpt", 
                            "value": prediction
                        }
                    ],
                    "image_abs": image_output_path,
                    "image": f"images/{image_filename}",
                    "original_question": data_dict['question'],
                    "original_answers": data_dict.get('answers', []),
                    "dataset_name": data_dict.get('dataset_name', ''),
                    "type": data_dict.get('type', ''),
                    "image_path": data_dict['image_path']
                }
                
                results.append(result_item)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # 保存结果
        output_json_path = os.path.join(output_dir, 'labeled_ocrbench_v2.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Labeled {len(results)} samples using fallback method")
        print(f"Results saved to: {output_json_path}")
        
        return results
        
    except Exception as e:
        print(f"Error in fallback method: {e}")
        return []

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Label OCRBench v2 dataset using VLLM or transformers")
    parser.add_argument("--model_path", type=str, default=QWEN25VL7B, 
                       help="Path to the model")
    # parser.add_argument("--json_data_file", type=str, default="OCRBench_v2_new_5.json",
    parser.add_argument("--json_data_file", type=str, default="OCRBench_v2.json",
                       help="OCRBench v2 JSON data file name")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (if None, uses default)")
    parser.add_argument("--use_fallback", action="store_true",
                       help="Force use transformers fallback method")
    parser.add_argument("--force_vllm", action="store_true", default=True,
                       help="Force use VLLM and fail if VLLM fails (default: True)")
    parser.add_argument("--allow_fallback", action="store_true",
                       help="Allow automatic fallback to transformers if VLLM fails")
    
    args = parser.parse_args()
    
    # 处理force_vllm逻辑
    if args.allow_fallback:
        force_vllm = False
    else:
        force_vllm = args.force_vllm
    
    print("Starting OCRBench v2 labeling...")
    print(f"Model: {args.model_path}")
    print(f"Data file: {args.json_data_file}")
    print(f"Output dir: {args.output_dir or 'auto-generated'}")
    print(f"Force VLLM: {force_vllm}")
    
    if args.use_fallback:
        print("Using transformers fallback method...")
        results = label_ocrv2_fallback(
            model_name_or_path=args.model_path,
            json_data_file=args.json_data_file,
            output_dir=args.output_dir
        )
    else:
        results = label_ocrv2(
            model_name_or_path=args.model_path,
            json_data_file=args.json_data_file,
            output_dir=args.output_dir,
            force_vllm=force_vllm
        )
    
    print(f"Labeling completed! Generated {len(results)} labeled samples.")
    
    # 示例用法说明：
    # 默认使用VLLM (推荐):
    # python label_ocrbench_v2.py --model_path /path/to/model
    # 
    # 允许自动fallback到transformers:
    # python label_ocrbench_v2.py --model_path /path/to/model --allow_fallback
    # 
    # 强制使用transformers方法:
    # python label_ocrbench_v2.py --model_path /path/to/model --use_fallback
    # 
    # 指定数据文件:
    # python label_ocrbench_v2.py --model_path /path/to/model --json_data_file OCRBench_v2_new_5.json