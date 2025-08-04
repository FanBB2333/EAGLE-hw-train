import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
sys.path.append('..')
import timeout_decorator
import signal
import multiprocessing
import argparse
import logging
from typing import Union
from tqdm import tqdm
import time
from pathlib import Path
import json
from accelerate import Accelerator
from accelerate.utils import gather_object
import pickle
import os
from transformers import AutoTokenizer
from safetensors.torch import safe_open
from train_video1 import ModelArguments

eval_logger = logging.getLogger("eval_video")
CURRENT_DIR = Path(__file__).resolve().parent

try:
    from eagle.model import *
    from eagle.model.builder import load_pretrained_model
    from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from eagle.conversation import conv_templates, SeparatorStyle
    from eagle.datasets.video_dataset import smart_tokenizer_and_embedding_resize
except ImportError:
    eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")
    raise ImportError("omg")

from fzy.ds import ActivityNetCaps, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2, MVBench
handle_stuck = False

# Available tasks
AVAILABLE_TASKS = ["activitynet", "breakfast", "charades", "qvhighlights", "valor", "youcook2", "mvbench"]

def validate_tasks(task_string):
    """Validate and parse task string into a list of valid tasks."""
    tasks = [task.strip() for task in task_string.split(',')]
    invalid_tasks = [task for task in tasks if task not in AVAILABLE_TASKS]
    if invalid_tasks:
        raise ValueError(f"Invalid task(s): {invalid_tasks}. Available tasks: {AVAILABLE_TASKS}")
    return tasks

def get_dataset(task):
    """Initialize dataset according to task name."""
    if task == "activitynet":
        return ActivityNetCaps()
    elif task == "breakfast":
        return Breakfast()
    elif task == "charades":
        return Charades()
    elif task == "qvhighlights":
        return QVHighlights()
    elif task == "valor":
        return VALOR32K()
    elif task == "youcook2":
        return YouCook2()
    elif task == "mvbench":
        return MVBench()
    else:
        raise NotImplementedError(f"Task {task} not implemented")

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old", 
        help="Pretrained path of model"
    )
    parser.add_argument(
        "--model_name", 
        default="eagle", 
        help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--task",
        default="charades",
        help="Task name(s) to evaluate. Can be a single task or multiple tasks separated by comma. "
             "Available tasks: activitynet, breakfast, charades, qvhighlights, valor, youcook2, mvbench",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--conv_template",
        default="llama3",
        help=("conv mode"),
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument('--distributed', action='store_true', help='Distributed evaluation')
    args = parser.parse_args()
    return args

def gen_prompt(data, args, task):
    question2 = None
    if task == "activitynet":
        duration = data["duration"]
        # question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
        # question2 = f'The video\'s duration is {duration}s. The event "{data["question"]}" starts at: '
        # question2 = "The event starts at 00:"
        # question1 = "What is the video about?"
        # question2 = "The video is about: "
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "charades":
        duration = data["answer"][1] - data["answer"][0]
        duration = float(f"{duration:.2f}")
        # question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video. The event starts at:'
        # question2 = f'The video\'s duration is {duration}s. The event "{data["question"]}" starts at: '
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "qvhighlights":
        duration = data["duration"]
        answers = data["answer"]
        # question1 = f"{data['question']}"
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task == "breakfast":
#{'User': f'The video lasts {duration:.1f} seconds. Please output the step-by-step actions the person is doing with start and end timestamps in the video.',
# 'Assistant': 'Based on the provided video, the step-by-step actions the person is doing with start and end timestamps in the video are:\nFrom 00:'}
        duration = data["duration"]
        question1 = f'The video lasts {duration:.1f} seconds. Please output the step-by-step actions the person is doing with start and end timestamps in the video.'
        question2 = f'Based on the provided video, the step-by-step actions the person is doing with start and end timestamps in the video are:\nFrom 00:'
    elif task in ["valor", "youcook2"]:
        duration = data["duration"]
        # question1 = f"{data['question']}"
        question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
    elif task in ["mvbench"]:
        question1 = data["question"]
    else:
        raise NotImplementedError(f"Task {task} not implemented")

    if DEFAULT_IMAGE_TOKEN not in question1:
        question1 = DEFAULT_IMAGE_TOKEN + '\n' + question1
    # args.conv_template: llama3
    conv = conv_templates[args.conv_template].copy()
    # 0: user, 1: assistant
    conv.append_message(conv.roles[0], question1)
    if question2 is not None:
        conv.append_message(conv.roles[1], question2)
        
    # conv.append_message(conv.roles[1], question2)
    # conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    return prompt_question


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")

def get_image_tensor(data, image_processor, model, args):
    try:
        image_tensor = process_images(
            images=data["data_path"],
            image_processor=image_processor,
            model_cfg=model.config,
            modality='video',
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
    except Exception as e:
        eval_logger.error(f"Error {e} in processing images")
        return None

# @timeout_decorator.timeout(3)
def get_image_tensor_timeout(data, image_processor, model, args, queue):
    try:
        image_tensor = process_images(
            images=data["data_path"],
            image_processor=image_processor,
            model_cfg=model.config,
            modality='video',
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
        queue.put(("success", image_tensor))
    except Exception as e:
        queue.put(("error", str(e)))


def load_video_model(args, modality='video'):
    """
    Load video model with tokenizer and vision processor
    
    Args:
        args: Arguments containing model_path and device
        modality: Model modality ('video' or 'audio')
    
    Returns:
        tuple: (model, tokenizer, image_processor, modality)
    """
    # base_path = "/home6/fzy/repos/EAGLE/model/LLM/Llama-3.2-3B-Instruct"
    base_path = "./model/LLM/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = EagleLlamaForCausalLM.from_pretrained(
        base_path,
        low_cpu_mem_usage=True,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="<pad>"),
        tokenizer=tokenizer,
        model=model,
    )
    
    model_args_path = os.path.join(args.model_path, "model_args.pkl")
    if os.path.exists(model_args_path):
        with open(model_args_path, 'rb') as f:
            model_args: ModelArguments = pickle.load(f)
            model_args.evaluation = True  # Set evaluation to True for the model

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp="",
        modality=modality
    )
    tensors = {}

    safetensor_paths = [
        os.path.join(args.model_path, "model-00001-of-00002.safetensors"),
        os.path.join(args.model_path, "model-00002-of-00002.safetensors"),
    ]
    
    for safetensor_path in safetensor_paths:
        with safe_open(safetensor_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    print(f"警告：键 {k} 在多个safetensor文件中出现，后面的值会覆盖前面的值")
                tensors[k] = f.get_tensor(k)
    
    model.load_state_dict(tensors, strict=False)
    model = model.to(torch.float16)
    model.cuda()

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor.video_processor

    model.eval()
    
    # Determine modality based on model path
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    
    print(f"Modality: {modality}, model type: {type(model)}, pretrained model: {args.model_path}")
    
    return model, tokenizer, image_processor, modality

            
@torch.no_grad()
def evaluate_single_task(args: Union[argparse.Namespace, None] = None, task: str = None) -> None:
    # Load video model using the extracted function
    model, tokenizer, image_processor, modality = load_video_model(args, modality='video')
    
    print(f"Task: {task}")
    # initialize dataset according to task
    ds = get_dataset(task)
    # return
    
    # time.sleep(100)
    # test_dataset = PointLLMDataset()

    test_dataloader = [{
        # "data_path": "/home6/fzy/repos/EAGLE/dataset/ActivityNetCaps/v1-2/val/v_ZMTi498qnPc.mp4",
        # "data_path": "/home6/fzy/repos/EAGLE/dataset/Charades/Charades_v1_480/0BNML.mp4",
        # "data_path": "/home6/fzy/repos/EAGLE/dataset/Charades/Charades_v1_480/0BZAD.mp4",
        "data_path": "/home6/fzy/repos/EAGLE/dataset/ActivityNetCaps/all_test/v__4S7eaL-cR8.mp4",
        "question": "What does the video show?",
        "answer": "xxx"
    }]
    test_dataloader = ds
        
    gen_list = list()
    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    # model.get_vision_tower().config.num_frames = 16
    for i, data in enumerate(test_dataloader):
        # data = data[0]

        if not handle_stuck:
            try:
                # print(f"Loading {data['idx']} data")
                image_tensor = process_images(
                    images=[data["data_path"]],
                    image_processor=image_processor,
                    model_cfg=model.config,
                    modality=modality,
                )
                image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
            except Exception as e:
                eval_logger.error(f"Error {e} in processing images")
                continue
        else:
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=get_image_tensor_timeout, args=(data, image_processor, model, args, queue))
            process.start()
            # 设置超时时间（秒）
            timeout = 5
            process.join(timeout)

            if process.is_alive():
                print("Timeout: The task took too long to complete.")
                process.terminate()  # 超时后强制终止进程
                continue
            else:
                # 获取子进程的返回值
                result, data = queue.get()
            if result == "success":
                image_tensor = data
            else:
                print("Error occurred:", data)
                continue
            if image_tensor is None:
                continue
        
        # Extract video_grid_thw if available
        if hasattr(image_tensor, "video_grid_thw"):
            video_grid_thw = image_tensor['video_grid_thw']
            image_tensor = image_tensor['pixel_values_videos']
        else:
            video_grid_thw = None
            print("No video_grid_thw found, using None for video_grid_thw")
            
        prompt_question = gen_prompt(data, args, task)
        # print(prompt_question)
        
        input_ids = tokenizer_image_token(
            prompt_question, 
            tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors="pt"
        )
        pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # print(input_ids)
        input_ids = pad_sequence(
            tokenizer=tokenizer,
            input_ids=[input_ids], 
            batch_first=True, 
            padding_value=pad_token_ids
        ).to(args.device)
        attention_masks = input_ids.ne(pad_token_ids).to(args.device)

        gen_kwargs = {}
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1

        try:
            pbar.update(1)
            cont = model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=args.use_cache,
                modality=modality,
                video_grid_thw=video_grid_thw
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            # print(text_outputs)
        except Exception as e:
            continue
        #     eval_logger.error(f"Error {e} in generating")
        #     cont = ""
        #     text_outputs = [""]
        gen_list.append({
            "task": task,
            **data,
            "prediction": text_outputs[0],
        })
        with open(str(CURRENT_DIR.parent / "output/3b" / f"{task}_output.json"), "w") as f:
            json.dump(gen_list, f, indent=4)
    pbar.close()


def load_video_model_dist(args, accelerator, modality='video'):
    """
    Load video model with tokenizer and vision processor for distributed evaluation
    
    Args:
        args: Arguments containing model_path and device
        accelerator: Accelerator instance for distributed training
        modality: Model modality ('video' or 'audio')
    
    Returns:
        tuple: (model, tokenizer, image_processor, modality)
    """
    base_path = "./model/LLM/Llama-3.2-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = EagleLlamaForCausalLM.from_pretrained(
        base_path,
        low_cpu_mem_usage=True,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="<pad>"),
        tokenizer=tokenizer,
        model=model,
    )
    
    model_args_path = os.path.join(args.model_path, "model_args.pkl")
    if os.path.exists(model_args_path):
        with open(model_args_path, 'rb') as f:
            model_args: ModelArguments = pickle.load(f)
            model_args.evaluation = True  # Set evaluation to True for the model
    
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp="",
        modality=modality
    )
    tensors = {}

    safetensor_paths = [
        os.path.join(args.model_path, "model-00001-of-00002.safetensors"),
        os.path.join(args.model_path, "model-00002-of-00002.safetensors"),
    ]
    
    for safetensor_path in safetensor_paths:
        with safe_open(safetensor_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    print(f"警告：键 {k} 在多个safetensor文件中出现，后面的值会覆盖前面的值")
                tensors[k] = f.get_tensor(k)
    
    model.load_state_dict(tensors, strict=False)
    model = model.to(torch.float16)
    
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor.video_processor

    model = accelerator.prepare(model)
    model.eval()
    
    # Determine modality based on model path
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    
    print(f"Modality: {modality}, model type: {type(model)}, pretrained model: {args.model_path}")
    
    return model, tokenizer, image_processor, modality


@torch.no_grad()
def evaluate_dist_single_task(args: Union[argparse.Namespace, None] = None, task: str = None) -> None:
    accelerator = Accelerator()
    
    # Load video model using the extracted function
    model, tokenizer, image_processor, modality = load_video_model_dist(args, accelerator, modality='video')
    
    print(f"Task: {task}")
    # initialize dataset according to task
    ds = get_dataset(task)
    test_dataloader = ds
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(test_dataloader) as batch:
        print(f"Generating {len(batch)} samples")
        results=dict(outputs=[])
        for data in tqdm(batch, desc="Generating samples"):
            try:
                image_tensor = process_images(
                    images=[data["data_path"]],
                    image_processor=image_processor,
                    model_cfg=model.config,
                    modality=modality,
                )
                image_tensor = image_tensor.to("cuda", dtype=torch.float16)
            except Exception as e:
                eval_logger.error(f"Error {e} in processing images")
                continue

            # Extract video_grid_thw if available
            if hasattr(image_tensor, "video_grid_thw"):
                video_grid_thw = image_tensor['video_grid_thw']
                image_tensor = image_tensor['pixel_values_videos']
            else:
                video_grid_thw = None
                print("No video_grid_thw found, using None for video_grid_thw")

            # question = data["question"]
            # answer = data["answer"]
            # duration = data["answer"][1] - data["answer"][0]
            # # format to .2f
            # duration = float(f"{duration:.2f}")
            
            # question = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video.'
            
            # if DEFAULT_IMAGE_TOKEN not in question:
            #     question = DEFAULT_IMAGE_TOKEN + '\n' + question
            # # args.conv_template: llama3
            # conv = conv_templates[args.conv_template].copy()
            # # 0: user, 1: assistant
            # conv.append_message(conv.roles[0], question)
            # conv.append_message(conv.roles[1], f'The video\'s duration is {duration}s. The event "{data["question"]}" starts at: ')

            # conv.append_message(conv.roles[0], question)
            # conv.append_message(conv.roles[1], None)
            
            # prompt_question = conv.get_prompt()
            
            prompt_question = gen_prompt(data, args, task)
            # print(prompt_question)
            # return
            input_ids = tokenizer_image_token(
                prompt_question, 
                tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            )
            pad_token_ids = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            # print(input_ids)
            input_ids = pad_sequence(
                tokenizer=tokenizer,
                input_ids=[input_ids], 
                batch_first=True, 
                padding_value=pad_token_ids
            ).to(args.device)
            attention_masks = input_ids.ne(pad_token_ids).to(args.device)

            gen_kwargs = {}
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            # try:
            cont = model.generate(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=args.use_cache,
                modality=modality,
                video_grid_thw=video_grid_thw
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            # print(text_outputs)
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     cont = ""
            #     text_outputs = [""]
            results["outputs"].append({
                "task": task,
                **data,
                "prediction": text_outputs[0],
            })
        results = [results]
    gathered = gather_object(results)
    with open(str(CURRENT_DIR.parent / "output" / f"{task}_output_dist.json"), "w") as f:
        json.dump(gathered, f, indent=4)


def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    """Evaluate multiple tasks sequentially."""
    # Parse and validate tasks
    tasks = validate_tasks(args.task)
    
    print(f"Evaluating tasks: {tasks}")
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Starting evaluation for task: {task}")
        print(f"{'='*50}")
        
        try:
            if args.distributed:
                evaluate_dist_single_task(args=args, task=task)
            else:
                evaluate_single_task(args=args, task=task)
            print(f"✓ Completed evaluation for task: {task}")
        except Exception as e:
            print(f"✗ Error evaluating task {task}: {e}")
            eval_logger.error(f"Error evaluating task {task}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("All tasks completed!")
    print(f"{'='*50}")

def evaluate_dist(args: Union[argparse.Namespace, None] = None) -> None:
    """Distributed evaluation for multiple tasks."""
    # Parse and validate tasks
    tasks = validate_tasks(args.task)
    
    print(f"Evaluating tasks in distributed mode: {tasks}")
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Starting distributed evaluation for task: {task}")
        print(f"{'='*50}")
        
        try:
            evaluate_dist_single_task(args=args, task=task)
            print(f"✓ Completed distributed evaluation for task: {task}")
        except Exception as e:
            print(f"✗ Error in distributed evaluation for task {task}: {e}")
            eval_logger.error(f"Error in distributed evaluation for task {task}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("All distributed tasks completed!")
    print(f"{'='*50}")

def evaluate_with_results(model_path: str, datasets=None):
    """
    Wrapper function for compatibility with eval_video_all.py
    
    Args:
        model_path: Path to the pretrained model
        datasets: List of dataset names to evaluate. If None, evaluates all available tasks.
    
    Returns:
        Dictionary containing evaluation results with status and metrics
    """
    try:
        # Create mock args object similar to parse_eval_args
        args = argparse.Namespace()
        args.model_path = model_path
        args.model_name = "eagle"
        args.device = "cuda"
        args.output_path = None  # Will use default
        args.conv_template = "llama3"
        args.use_cache = None
        args.batch_size = 1
        args.gen_kwargs = ""
        args.model_args = ""
        args.distributed = False
        
        # Determine which tasks to run
        if datasets is None or len(datasets) == 0:
            # Run all available tasks
            tasks_to_run = AVAILABLE_TASKS.copy()
        else:
            # Validate and filter requested datasets
            tasks_to_run = []
            for dataset in datasets:
                if dataset.lower() in AVAILABLE_TASKS:
                    tasks_to_run.append(dataset.lower())
                else:
                    print(f"Warning: Dataset '{dataset}' not available in eval_video_qwen. Available: {AVAILABLE_TASKS}")
        
        if not tasks_to_run:
            return {
                'status': 'error',
                'error': f'No valid tasks to run. Available tasks: {AVAILABLE_TASKS}'
            }
        
        # Store original task arg
        args.task = ','.join(tasks_to_run)
        
        print(f"Starting eval_video_qwen evaluation for tasks: {tasks_to_run}")
        print(f"Model path: {model_path}")
        print("-" * 50)
        
        # Results container
        results = {}
        successful_tasks = []
        failed_tasks = []
        
        # Run evaluation for each task
        for task in tasks_to_run:
            print(f"\nEvaluating task: {task}")
            try:
                # Set current task
                args.task = task
                
                # Run single task evaluation
                evaluate_single_task(args=args, task=task)
                
                # Check if output file was created and read results
                output_dir = CURRENT_DIR.parent / "output" / "3b"
                output_file = output_dir / f"{task}_output.json"
                
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        task_results = json.load(f)
                    
                    # Process results for this task
                    task_metrics = process_task_results(task, task_results)
                    results[task] = task_metrics
                    successful_tasks.append(task)
                    
                    print(f"✓ Completed {task}: {len(task_results)} predictions generated")
                else:
                    print(f"✗ Output file not found for {task}: {output_file}")
                    failed_tasks.append(task)
                    results[task] = {
                        'status': 'error',
                        'error': 'Output file not generated'
                    }
                    
            except Exception as e:
                print(f"✗ Error evaluating {task}: {str(e)}")
                failed_tasks.append(task)
                results[task] = {
                    'status': 'error', 
                    'error': str(e)
                }
        
        # Compile final results
        final_results = {
            'status': 'completed',
            'model_path': model_path,
            'tasks_requested': tasks_to_run,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'total_tasks': len(tasks_to_run),
            'success_rate': len(successful_tasks) / len(tasks_to_run) if tasks_to_run else 0,
            'detailed_results': results
        }
        
        # Add summary statistics
        summary_stats = {}
        for task, task_data in results.items():
            if isinstance(task_data, dict) and 'total_predictions' in task_data:
                summary_stats[task] = {
                    'total_predictions': task_data['total_predictions'],
                    'output_file': task_data.get('output_file')
                }
        
        if summary_stats:
            final_results['summary_statistics'] = summary_stats
        
        print(f"\n📊 Evaluation Summary:")
        print(f"   - Total tasks: {len(tasks_to_run)}")
        print(f"   - Successful: {len(successful_tasks)}")
        print(f"   - Failed: {len(failed_tasks)}")
        print(f"   - Success rate: {final_results['success_rate']:.2%}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Critical error in eval_video_qwen: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e),
            'model_path': model_path
        }

def process_task_results(task: str, task_results):
    """
    Process raw task results and extract relevant metrics
    
    Args:
        task: Name of the task
        task_results: List of prediction results
    
    Returns:
        Dictionary with processed metrics
    """
    if not task_results:
        return {
            'status': 'error',
            'error': 'No results generated'
        }
    
    # Basic statistics
    total_predictions = len(task_results)
    
    # Extract sample predictions for inspection
    sample_predictions = []
    for i, result in enumerate(task_results[:5]):  # First 5 samples
        sample = {
            'index': i,
            'question': result.get('question', 'N/A'),
            'prediction': result.get('prediction', 'N/A'),
            'data_path': result.get('data_path', 'N/A')
        }
        
        # Add task-specific fields
        if 'answer' in result:
            sample['answer'] = result['answer']
        if 'duration' in result:
            sample['duration'] = result['duration']
            
        sample_predictions.append(sample)
    
    # Determine output file path
    output_dir = CURRENT_DIR.parent / "output" / "3b"
    output_file = str(output_dir / f"{task}_output.json")
    
    return {
        'status': 'success',
        'total_predictions': total_predictions,
        'output_file': output_file,
        'sample_predictions': sample_predictions,
        'task_specific_info': get_task_specific_info(task, task_results)
    }

def get_task_specific_info(task: str, task_results):
    """
    Extract task-specific information and statistics
    
    Args:
        task: Name of the task
        task_results: List of prediction results
    
    Returns:
        Dictionary with task-specific metrics
    """
    info = {'task': task}
    
    if not task_results:
        return info
    
    # Sample result for analysis
    sample = task_results[0] if task_results else {}
    
    if task == "activitynet":
        info['description'] = "ActivityNet Captions - Video captioning and temporal localization"
        # Extract duration statistics if available
        durations = [r.get('duration', 0) for r in task_results if 'duration' in r]
        if durations:
            info['avg_duration'] = sum(durations) / len(durations)
            info['max_duration'] = max(durations)
            info['min_duration'] = min(durations)
    
    elif task == "breakfast":
        info['description'] = "Breakfast Actions - Step-by-step action recognition"
        # Count action predictions
        action_count = len([r for r in task_results if 'prediction' in r and len(r['prediction']) > 0])
        info['predictions_with_actions'] = action_count
    
    elif task == "charades":
        info['description'] = "Charades Actions - Action localization and description"
        # Extract answer format info
        answers_with_timing = len([r for r in task_results if 'answer' in r and isinstance(r['answer'], list)])
        info['answers_with_timing'] = answers_with_timing
    
    elif task == "qvhighlights":
        info['description'] = "QV Highlights - Query-based video highlight detection"
    
    elif task == "valor":
        info['description'] = "VALOR32K - Video and language understanding"
    
    elif task == "youcook2":
        info['description'] = "YouCook2 - Instructional video understanding"
    
    elif task == "mvbench":
        info['description'] = "MVBench - Multi-view video understanding benchmark"
    
    return info

if __name__ == "__main__":
    args = parse_eval_args()
    print(args)
    if args.distributed:
        evaluate_dist(args=args)
    else:
        evaluate(args=args)
