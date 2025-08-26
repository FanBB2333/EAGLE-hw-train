"""
OCRBench v2 数据集标注工具

该脚本用于使用多模态大语言模型对OCRBench v2数据集进行自动标注，生成训练数据格式。

主要功能：
1. 使用VLLM、transformers或API加载多模态模型
2. 对OCRBench v2数据集中的图片和问题进行推理
3. 生成标准的对话格式训练数据
4. 复制并组织图片文件

支持的标注方法：
- vllm: 使用VLLM框架加载模型进行推理（推荐）
- transformers: 使用transformers库直接加载模型
- api: 使用预配置的VLLM API服务器（端口58000）

支持的数据文件：
- OCRBench_v2.json: 原始数据
- OCRBench_v2_new_5.json: 精选5k样本
- OCRBench_v2_new_all.json: 扩展数据集

输出格式：
- labeled_ocrbench_v2.json: 标注结果 
- images/: 复制的图片文件
- labeling_stats.json: 统计信息

使用方法：
python label_ocrbench_v2.py --method vllm --model_path /path/to/model
python label_ocrbench_v2.py --method api --json_data_file OCRBench_v2_new_5.json
"""
try:
    import vllm
except Exception as e:
    print(f"Warning: vllm import failed: {e}. VLLM-based labeling will not work.")
import openai
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
import shutil
import torch
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# 检测 bfloat16 支持
def check_bfloat16_support():
    """检测当前设备是否支持 bfloat16"""
    try:
        if torch.cuda.is_available():
            # 检查CUDA版本和GPU架构
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            
            # Ampere架构 (8.x) 及以上支持 bfloat16
            if capability[0] >= 8:
                # 进一步测试是否真的可以使用 bfloat16
                test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                return True
            else:
                print(f"GPU capability {capability} < 8.0, bfloat16 not supported")
                return False
        else:
            # CPU 也可能支持 bfloat16，但不如 GPU 常见
            test_tensor = torch.tensor([1.0], dtype=torch.bfloat16)
            return True
    except Exception as e:
        print(f"bfloat16 test failed: {e}")
        return False

# 全局变量：检测支持的数据类型
SUPPORTS_BFLOAT16 = check_bfloat16_support()
PREFERRED_DTYPE = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16

print(f"Data type support check: bfloat16={SUPPORTS_BFLOAT16}, using {PREFERRED_DTYPE}")

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
        from PIL import Image
        import torch
        
        # 获取可用GPU数量，限制使用4张
        gpu_count = min(torch.cuda.device_count(), 4)  # 最多使用4张GPU
        print(f"Available GPUs: {torch.cuda.device_count()}, Using: {gpu_count}")
        
        # 更保守的VLLM参数设置，避免内存问题
        vllm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": gpu_count,
            "trust_remote_code": True,
            "max_model_len": 1024,  # 降低模型长度减少内存使用
            "gpu_memory_utilization": 0.70,  # 降低GPU内存使用率
            "swap_space": 2,  # 减少swap空间
            "disable_custom_all_reduce": True,  # 禁用自定义all_reduce，避免通信问题
        }
        
        # 对于已知的多模态模型添加特定配置
        if any(name in model_name_or_path.lower() for name in ["qwen", "llava", "eagle"]):
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 1}
        
        llm = LLM(**vllm_kwargs)
        
        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,  # 减少生成长度以降低内存压力
            stop=None
        )
        
        print("VLLM model loaded successfully!")
        
        # 清理GPU内存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU memory cache after model loading")
        
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
    
    # 准备输入数据 - 修改为支持多模态
    inputs = []
    valid_indices = []
    
    for i, data_dict in enumerate(json_data):
        # 构建问题
        question = data_dict['question']
        # 添加提示词以提高回答质量
        question_with_prompt = question + '\nAnswer the question using a single word or phrase.'
        
        # 检查图片是否存在
        image_path = os.path.join(img_dir, data_dict['image_path'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"Preparing input {i+1}/{len(json_data)}: {data_dict['image_path']}, size: {image.size}")
            
            vllm_prompt = f"<image>\n{question_with_prompt}\n"
            
            # 构建VLLM官方格式的输入
            vllm_input = {
                "prompt": vllm_prompt,
                "multi_modal_data": {"image": image},
            }
            
            inputs.append({
                "vllm_input": vllm_input,
                "data_index": i
            })
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error preparing input for sample {i}: {e}")
            continue
    
    print(f"Prepared {len(inputs)} valid inputs for inference...")
    
    # 批量推理
    try:
        print("Starting VLLM multimodal inference using official format...")
        print("Example input format:")
        if inputs:
            example_input = inputs[0]["vllm_input"]
            print(f"  Prompt: {example_input['prompt'][:100]}...")
            print(f"  Image: PIL.Image {example_input['multi_modal_data']['image'].size}")
        
        # 分批处理以避免内存问题，提高GPU利用率
        if gpu_count >= 4:
            batch_size = 4   # 大幅减少批大小
        elif gpu_count >= 2:
            batch_size = 2   # 2张GPU，小批次
        else:
            batch_size = 1   # 单GPU，逐个处理
        
        print(f"Using conservative VLLM batch size: {batch_size} for {gpu_count} GPUs")
        all_outputs = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            print(f"Processing VLLM batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
            
            try:
                # 在每批处理前清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 使用VLLM官方的多模态批量推理格式
                batch_vllm_inputs = [inp["vllm_input"] for inp in batch_inputs]
                
                # VLLM官方批量推理方法
                batch_outputs = llm.generate(batch_vllm_inputs, sampling_params)
                all_outputs.extend(batch_outputs)
                
            except Exception as batch_error:
                print(f"Error in batch {i//batch_size + 1}: {batch_error}")
                print("Trying to process batch items individually...")
                
                # 如果批处理失败，尝试逐个处理
                for inp in batch_inputs:
                    try:
                        # 使用VLLM官方的单个输入格式
                        single_output = llm.generate([inp["vllm_input"]], sampling_params)
                        all_outputs.extend(single_output)
                    except Exception as single_error:
                        print(f"Error processing single item: {single_error}")
                        # 创建空输出以保持索引一致
                        all_outputs.append(None)
        
        outputs = all_outputs
        
        # 处理输出
        for i, (inp, output) in enumerate(zip(inputs, outputs)):
            data_index = inp["data_index"]
            data_dict = json_data[data_index]
            
            # 处理可能的None输出
            if output is None:
                prediction = ""
                print(f"Warning: Failed to process sample {data_index}")
            else:
                # 提取VLLM chat输出
                if hasattr(output, 'outputs') and output.outputs and len(output.outputs) > 0:
                    prediction = output.outputs[0].text.strip()
                elif hasattr(output, 'content') and output.content:
                    prediction = output.content.strip()
                else:
                    prediction = str(output).strip() if output else ""
                    print(f"Warning: Unexpected output format for sample {data_index}")
            
            if not prediction:
                print(f"Warning: Empty output for sample {data_index}")
            
            # 复制图片到输出目录
            image_filename = f"{data_dict['id']}_{os.path.basename(data_dict['image_path'])}"
            image_output_path = os.path.join(images_output_dir, image_filename)
            image_source_path = os.path.join(img_dir, data_dict['image_path'])
            shutil.copy2(image_source_path, image_output_path)
            
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
        print(f"Error during VLLM inference: {e}")
        
        # 检查是否是多模态相关的错误
        if "image" in str(e).lower() or "multimodal" in str(e).lower() or "unknown part type" in str(e).lower():
            print("Detected multimodal incompatibility with current VLLM version.")
            print("VLLM may not support multimodal inference for this model.")
            
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
    使用transformers直接加载Qwen2.5-VL模型对OCRBench v2数据集进行标注
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    
    print("Loading Qwen2.5-VL model using transformers...")
    
    # 检查可用GPU
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    # 根据GPU数量选择加载策略
    if gpu_count > 1:
        print(f"Using device_map='auto' for multi-GPU setup with {gpu_count} GPUs")
        device_map = "auto"
    else:
        print("Using single GPU")
        device_map = "cuda:0"
    
    # 直接加载Qwen2.5-VL模型，使用更保守的参数避免OOM
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=PREFERRED_DTYPE,  # 使用检测到的最佳数据类型
        device_map=device_map,
        low_cpu_mem_usage=True,  # 减少CPU内存使用
        # max_memory可以手动指定每个GPU的最大使用量
        # max_memory={0: "20GB", 1: "20GB"} if gpu_count > 1 else None
    )
    
    # 加载processor，使用默认设置以保持图片原始大小
    # 不设置min_pixels和max_pixels，让模型使用原始图片分辨率
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    model.eval()
    
    # 启用梯度检查点以进一步减少显存（如果支持）
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing to save memory")
    
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
    
    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    for i, data_dict in enumerate(tqdm(json_data, desc="Labeling with Qwen2.5-VL")):
        # 每处理一定数量的样本后清理显存
        if i > 0 and i % 10 == 0:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                print(f"Sample {i}, GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # 加载图片并保持原始大小
        image_path = os.path.join(img_dir, data_dict['image_path'])
        if not os.path.exists(image_path):
            continue
        
        image = Image.open(image_path).convert('RGB')
        print(f"Processing image {i+1}/{len(json_data)}: {data_dict['image_path']}, size: {image.size}")
        
        # 构建问题 - Qwen2.5-VL格式
        question = data_dict['question'] + '\nAnswer the question using a single word or phrase.'
        
        # 使用Qwen2.5-VL的消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # 处理输入 - 使用正确的process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate - 使用更保守的参数减少显存使用
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,  # 减少生成长度
                do_sample=False,
                temperature=0.0,
                use_cache=True,  # 启用KV cache
                pad_token_id=processor.tokenizer.eos_token_id,
                # 可以添加以下参数进一步优化显存
                # num_beams=1,  # 使用beam search=1
                # early_stopping=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode
        prediction = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
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
    
    # 保存结果
    output_json_path = os.path.join(output_dir, 'labeled_ocrbench_v2.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Labeled {len(results)} samples using fallback method")
    print(f"Results saved to: {output_json_path}")
    
    return results

def check_vllm_server_status(base_url: str = "http://localhost:58000"):
    """检查VLLM服务器状态"""
    import requests
    try:
        # 检查健康状态
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ VLLM server is running at {base_url}")
            return True
        else:
            print(f"❌ VLLM server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to VLLM server at {base_url}: {e}")
        print("Please start the VLLM server first:")
        print("  ./start_vllm_server.sh")
        return False

def label_ocrv2_api(model_name_or_path: str = "Qwen2.5-VL-7B-Instruct", json_data_file: str = "OCRBench_v2.json", output_dir: str = None):
    """
    使用OpenAI API对OCRBench v2数据集进行标注
    
    Args:
        model_name_or_path: 模型名称（用于API调用和输出目录命名）
        json_data_file: OCRBench v2数据文件名
        output_dir: 输出目录，如果为None则使用默认目录
    """
    import openai
    import base64
    from io import BytesIO
    
    print(f"Using OpenAI API for OCRBench v2 labeling with model: {model_name_or_path}")
    
    # 首先检查VLLM服务器状态
    base_url = "http://localhost:58000/v1"
    server_base = "http://localhost:58000"
    
    if not check_vllm_server_status(server_base):
        raise Exception("VLLM server is not available. Please start it first.")
    
    # 设置OpenAI客户端 - 使用预配置的VLLM服务器
    api_key = "EMPTY"  # VLLM服务器通常使用"EMPTY"作为API密钥
    
    # 创建OpenAI客户端实例
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    print(f"Using VLLM API server at: {base_url}")
    print(f"Model name: {model_name_or_path}")
    
    # 测试连接
    try:
        models = client.models.list()
        print(f"Available models: {[model.id for model in models.data]}")
    except Exception as e:
        print(f"Warning: Failed to connect to VLLM server: {e}")
        print("Please make sure VLLM server is running on port 58000")
        print("Start server with: ./start_vllm_server.sh")
    
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
    
    def encode_image_base64(image_path):
        """将图片编码为base64格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # 进行推理
    results = []
    
    print(f"Processing {len(json_data)} samples using API...")
    
    for i, data_dict in enumerate(tqdm(json_data, desc="Labeling with API")):
        # 检查图片是否存在
        image_path = os.path.join(img_dir, data_dict['image_path'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            print(f"Processing image {i+1}/{len(json_data)}: {data_dict['image_path']}, size: {image.size}")
            
            # 构建问题
            question = data_dict['question'] + '\nAnswer the question using a single word or phrase.'
            
            # 将图片编码为base64
            base64_image = encode_image_base64(image_path)
            
            # 构建API请求消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            response = client.chat.completions.create(
                model=model_name_or_path,
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )
            
            # 提取预测结果
            prediction = response.choices[0].message.content.strip() if response.choices else ""
            
            if not prediction:
                print(f"Warning: Empty response for sample {i}")
                prediction = ""
            
            # 复制图片到输出目录
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
    
    print(f"Labeled {len(results)} samples using API method")
    print(f"Results saved to: {output_json_path}")
    print(f"Images saved to: {images_output_dir}")
    
    # 保存统计信息
    stats = {
        "total_samples": len(json_data),
        "labeled_samples": len(results),
        "model_name": model_name_or_path,
        "json_data_file": json_data_file,
        "timestamp": datetime.now().isoformat(),
        "method": "api"
    }
    
    stats_path = os.path.join(output_dir, 'labeling_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return results
    


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Label OCRBench v2 dataset using different methods")
    parser.add_argument("--model_path", type=str, default=QWEN25VL7B, 
                       help="Path to the model (for vllm and transformers methods)")
    parser.add_argument("--json_data_file", type=str, default="OCRBench_v2.json",
                       help="OCRBench v2 JSON data file name")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (if None, uses default)")
    parser.add_argument("--method", type=str, default="vllm", choices=["vllm", "transformers", "api"],
                       help="Method to use for labeling: vllm, transformers, or api")
    
    args = parser.parse_args()
    
    print("Starting OCRBench v2 labeling...")
    print(f"Method: {args.method}")
    print(f"Model: {args.model_path}")
    print(f"Data file: {args.json_data_file}")
    print(f"Output dir: {args.output_dir or 'auto-generated'}")
    
    if args.method == "api":
        print("Using OpenAI API method with VLLM server...")
        print("Note: Make sure VLLM server is running on port 58000")
        print("Start server with: ./start_vllm_server.sh")
        print("Test connection with: python test_vllm_api.py")
        
        try:
            results = label_ocrv2_api(
                model_name_or_path="Qwen2.5-VL-7B-Instruct",  # 使用VLLM服务器上的模型名
                json_data_file=args.json_data_file,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"❌ API method failed: {e}")
            print("\nTroubleshooting steps:")
            print("1. Check if VLLM server is running: ./start_vllm_server.sh")
            print("2. Test API connection: python test_vllm_api.py")
            print("3. Check server logs for errors")
            exit(1)
    elif args.method == "transformers":
        print("Using transformers method...")
        results = label_ocrv2_fallback(
            model_name_or_path=args.model_path,
            json_data_file=args.json_data_file,
            output_dir=args.output_dir
        )
    else:  # vllm
        print("Using VLLM method...")
        results = label_ocrv2(
            model_name_or_path=args.model_path,
            json_data_file=args.json_data_file,
            output_dir=args.output_dir,
            force_vllm=True
        )
    
    print(f"Labeling completed! Generated {len(results)} labeled samples.")
    
    # 示例用法说明：
    # 使用VLLM方法 (默认):
    # python label_ocrbench_v2.py --method vllm --model_path /path/to/model
    # 
    # 使用transformers方法:
    # python label_ocrbench_v2.py --method transformers --model_path /path/to/model
    # 
    # 使用API方法:
    # python label_ocrbench_v2.py --method api
    # 
    # 指定数据文件:
    # python label_ocrbench_v2.py --method vllm --json_data_file OCRBench_v2_new_5.json