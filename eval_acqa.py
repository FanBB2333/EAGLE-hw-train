import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import sys
sys.path.append('./')
import json

import argparse
import logging
from typing import Union
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import copy
import pickle

eval_logger = logging.getLogger("eval_video")

# try:
from eagle.model import *
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.datasets.video_dataset import make_supervised_data_module, smart_tokenizer_and_embedding_resize, DataArguments
from eagle.conversation import conv_templates, SeparatorStyle
# import safe load
from safetensors.torch import safe_open
from train_video1 import ModelArguments
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/finetune/pr_llm/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old", 
        help="Pretrained path of model"
    )
    parser.add_argument(
        "--model_name", 
        default="eagle", 
        help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--tasks",
        default=None,
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
        default='/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/fzy/output/acqa.json',
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
    args = parser.parse_args()
    return args

import torch
from torch.utils.data import Dataset

import json
import os

import os
from dataclasses import dataclass

@dataclass
class VQADataInput:
    data_path: str | os.PathLike
    question: str
    answer: str

class VQADataset(Dataset):
    def __init__(self):
        super().__init__()
        self.json_data = json.load(open('/home1/hxl/disk2/Backup/EAGLE/chenxn/OCRBench_v2/OCRBench_v2.json'))
        self.img_dir = '/home1/hxl/disk2/Backup/EAGLE/chenxn/OCRBench_v2'
        self.add_prompt = True
    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index) -> VQADataInput:
        data_dict = self.json_data[index]
        if self.add_prompt:
            question = data_dict['question'] + '\nAnswer the question using a single word or phrase.'
        else:
            question = data_dict['question']
        answer = data_dict['answers'][0]
        return VQADataInput(
            data_path=os.path.join(self.img_dir, data_dict['image_path']),
            question=question,
            answer=answer,
        )
    def collate_fn(self, input):
        return input

def custom_collate_fn(batch):
    # 假设你想做一些处理，比如将所有数据拼接到一起
    return batch  # 修改为你自己的拼接逻辑


video_base = "/home1/hxl/disk2/EAGLE/.cache/huggingface/activitynetqa/all_test"
def activitynetqa_doc_to_visual(doc):
    video_path = os.path.join(video_base, f"v_{doc['video_name']}.mp4")
    extensions = ["mp4", "webm", "mkv"]
    for ext in extensions:
        modified_path = video_path.replace("mp4", ext)
        if os.path.exists(modified_path):
            return [modified_path]
    sys.exit(f"video path:{video_path} does not exist, please check")
    
@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    modality = 'video'
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
            model_args = pickle.load(f)
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp="",
        modality = 'video'
    )
    tensors = {}

    safetensor_paths = [
        os.path.join(args.model_path, "model-00001-of-00002.safetensors"),
        os.path.join(args.model_path, "model-00002-of-00002.safetensors"),
        # "checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle/model-00001-of-00002.safetensors",
        # "checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle/model-00002-of-00002.safetensors",
    ]
    
    for safetensor_path in safetensor_paths:
        with safe_open(safetensor_path, framework="pt", device='cpu') as f:
            for k in f.keys():
                if k in tensors:
                    print(f"警告：键 {k} 在多个safetensor文件中出现，后面的值会覆盖前面的值")
                # print(k)
                # if 'vision_tower' in k:
                #     continue
                tensors[k] = f.get_tensor(k)
    
    # tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items() if v is not None}
    model.load_state_dict(tensors, strict=False)
    # change model dtype to bfloat16
    model = model.to(torch.bfloat16)
    model.cuda()
    # 替换模型参数
    # for name, param in model.named_parameters():
    #     if "vision_tower" not in name:
    #         if 'mm_projector' in name:
    #             print(1)
    #         if name in tensors.keys():
    #             if tensors[name].shape == param.data.shape:
    #                 original_requires_grad = copy.deepcopy(param.requires_grad)  # 保存原始的requires_grad状态
    #                 original_device = copy.deepcopy(param.device)
    #                 original_dtype = copy.deepcopy(param.dtype)
    #                 param.data.copy_(tensors[name])
    #                 param.requires_grad = original_requires_grad  # 恢复requires_grad状态
    #                 param = param.to(original_device, dtype=original_dtype)
    #                 print(f"Loaded {name} from safetensor with shape {tensors[name].shape} to model with shape {param.data.shape}")
    #             else:
    #                 print(f"Shape mismatch for {name}: {tensors[name].shape} != {param.data.shape}")
    #         else:
    #             print(f"Key {name} not found in safetensor")

    # projector_path = "/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Video/pretrain/pretrain-video-llama3.2-3b-fzy-qwen2vl-eagle/mm_projector.bin"
    # mm_projector_weights = torch.load(projector_path, map_location='cpu')
    # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    # model.load_state_dict(mm_projector_weights, strict=False)

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor.video_processor

    # load state dict from args.model_path
    # tokenizer, model, image_processor, max_length = load_pretrained_model(
    #     model_path=args.model_path,
    #     model_base=None,
    #     model_name=args.model_name,
    #     modal=modality
    # )
    
    # image_processor = image_processor.video_processor
    print(f"image processor: {type(image_processor)}")
    model.eval()
    test_dataset = load_dataset("lmms-lab/ActivityNetQA")['test']
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=custom_collate_fn
    )

    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    outputs = []
    for i, data in enumerate(test_dataloader):
        data = data[0]
        video_file = activitynetqa_doc_to_visual(data)
        image_tensor = process_images(
            images=video_file,
            image_processor=image_processor,
            model_cfg=model.config,
            modality=modality,
        )
        image_tensor = image_tensor.to(dtype=torch.bfloat16, device=args.device)
                # image = image_full['pixel_values_videos']
                # video_grid_thw = image_full['video_grid_thw']
        if hasattr(image_tensor, "video_grid_thw"):
            video_grid_thw = image_tensor['video_grid_thw']
            image_tensor = image_tensor['pixel_values_videos']
        else:
            video_grid_thw = None
            print("No video_grid_thw found, using None for video_grid_thw")
        print(f"data: {data}")
        question = data['question']
        answer = data['answer']

        # DEFAULT_POINT_TOKEN 是点云的，视频的可能需要重写，可以参考如下方式修改prompt
        if DEFAULT_IMAGE_TOKEN not in question:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
        
        conv = conv_templates[args.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

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
            gen_kwargs["max_new_tokens"] = 20
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        # print(f"image_tensor: {image_tensor}")
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
        outputs.append(
            {
                "question": question,
                "answer": answer,
                "prediction": text_outputs
            }
        )
        print(outputs[-1])
        pbar.update(1)
    pbar.close()
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    # with open(args.output_path, 'w') as output_file:
    #     json.dump(outputs, output_file)
    
    print("Save at", args.output_path)


def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def eval_res(args: Union[argparse.Namespace, None] = None):
    from word2number import w2n
    from num2words import num2words
    def equal(pred, gt):
        if pred.lower() in gt.lower():
            return True
        if gt.lower() in pred.lower():
            return True
        # if gt is a number, convert to string and compare
        try:
            # gt_num = w2n.word_to_num(gt.lower())
            gt_str = num2words(gt, lang='en')
            if gt_str.lower() in pred.lower() or pred.lower() in gt_str.lower():
                return True
        except Exception as e:
            pass
        extend_dict = {
            "1": ["a", "one", "1st", "first"],
            "2": [ "two", "2nd", "second"],
            "0": ["zero", "0th", "zeroth"],
        }
        for k, v in extend_dict.items():
            if k in gt.lower() and any(x in pred.lower() for x in v):
                return True
            if k in pred.lower() and any(x in gt.lower() for x in v):
                return True
        return False
                
        
    output_file = args.output_path
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist, please check")
        return
    with open(output_file, 'r') as f:
        outputs = json.load(f)
    print(f"Loaded {len(outputs)} outputs from {output_file}")
    scores = list()
    for output in outputs:
        question = output['question']
        answer = output['answer']
        prediction = output['prediction'][0]
        if equal(prediction, answer):
            scores.append(1)
        else:
            scores.append(0)
    print(f"Accuracy: {sum(scores) / len(scores) * 100:.2f}%")
    print(f"Total: {len(scores)}, Correct: {sum(scores)}")
    
    

if __name__ == "__main__":
    args = parse_eval_args()
    # evaluate(args=args)
    eval_res(args=args)
