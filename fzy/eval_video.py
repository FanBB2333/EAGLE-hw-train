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

eval_logger = logging.getLogger("eval_video")
CURRENT_DIR = Path(__file__).resolve().parent

try:
    from eagle.model.builder import load_pretrained_model
    from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from eagle.conversation import conv_templates, SeparatorStyle
except ImportError:
    eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")
    raise ImportError("omg")

from eval.dataset.pointllm import PointLLMDataset
from eval.utils import DEFAULT_POINT_TOKEN

from fzy.ds import ActivityNetCaps, Breakfast, Charades, QVHighlights, VALOR32K, YouCook2
handle_stuck = False

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/video_finetune_1epoch", 
        help="Pretrained path of model"
    )
    parser.add_argument(
        "--model_name", 
        default="eagle", 
        help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--task",
        default="youcook2",
        choices=["activitynet", "breakfast", "charades", "qvhighlights", "valor", "youcook2"],
        help="To get full list of tasks, use the command lmms-eval --tasks list",
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

def gen_prompt(data, args):
    task = args.task
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
    # conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    return prompt_question


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out!")

def get_image_tensor(data, image_processor, model, args):
    try:
        image_tensor = process_images(
            images=data["data_path"],
            image_processor=image_processor,
            model_cfg=model.config
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
            model_cfg=model.config
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
        queue.put(("success", image_tensor))
    except Exception as e:
        queue.put(("error", str(e)))
        
            
@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    accelerator = Accelerator()
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        distributed=accelerator,
    )
    model.eval()
    modality = 'image'
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    print(f"Modality: {modality}, model type: {type(model)}, pretrained model: {args.model_path}, task: {args.task}")
    # initialize dataset according to args.task
    task = args.task
    if task == "activitynet":
        ds = ActivityNetCaps()
    elif task == "breakfast":
        ds = Breakfast()
    elif task == "charades":
        ds = Charades()
    elif task == "qvhighlights":
        ds = QVHighlights()
    elif task == "valor":
        ds = VALOR32K()
    elif task == "youcook2":
        ds = YouCook2()
        # handle_stuck = True
    else:
        raise NotImplementedError(f"Task {task} not implemented")
    # return
    
    model.cuda()
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
    # test_dataloader = DataLoader(
    #     ds,
    #     collate_fn=ds.collate_fn,
    #     batch_size=1,
    #     shuffle=False,
    # )
        
    gen_list = list()
    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    # model.get_vision_tower().config.num_frames = 16
    for i, data in enumerate(test_dataloader):
        # data = data[0]

        if not handle_stuck:
            try:
                # print(f"Loading {data['idx']} data")
                image_tensor = process_images(
                    images=data["data_path"],
                    image_processor=image_processor,
                    model_cfg=model.config
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
        prompt_question = gen_prompt(data, args)
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


@torch.no_grad()
def evaluate_dist(args: Union[argparse.Namespace, None] = None) -> None:
    accelerator = Accelerator()
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        distributed=accelerator,
    )
    model.eval()
    modality = 'image'
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    print(f"Modality: {modality}, model type: {type(model)}, pretrained model: {args.model_path}, task: {args.task}")
    # initialize dataset according to args.task
    task = args.task
    if task == "activitynet":
        ds = ActivityNetCaps()
    elif task == "breakfast":
        ds = Breakfast()
    elif task == "charades":
        ds = Charades()
    elif task == "qvhighlights":
        ds = QVHighlights()
    else:
        raise NotImplementedError(f"Task {task} not implemented")
    test_dataloader = ds.data
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(test_dataloader) as batch:
        print(f"Generating {len(batch)} samples")
        results=dict(outputs=[])
        for data in tqdm(batch, desc="Generating samples"):
            image_tensor = process_images(
                images=data["data_path"],
                image_processor=image_processor,
                model_cfg=model.config
            )
            image_tensor = image_tensor.to("cuda", dtype=torch.float16)

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
            
            prompt_question = gen_prompt(data, args)
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

if __name__ == "__main__":
    args = parse_eval_args()
    print(args)
    if args.distributed:
        evaluate_dist(args=args)
    else:
        evaluate(args=args)
