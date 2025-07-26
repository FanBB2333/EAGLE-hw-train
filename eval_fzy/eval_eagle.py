import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
import json
from datetime import datetime

import argparse
import logging
from typing import Union
from tqdm import tqdm
from PIL import Image

eval_logger = logging.getLogger("eval_eagle")

# try:
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")

from eval.utils import DEFAULT_POINT_TOKEN

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle/checkpoint-30000", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-900",
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen25vl-batch-llava-eagle/",
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-epoch2",
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc",
        default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc2",
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
        default='all_inc.json',
        type=str,
        metavar="= [filename.json]",
        help="The output filename (e.g., all_5.json). The file will be saved in a date-based directory structure automatically.",
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
        "--json_data",
        default="OCRBench_v2_new_all.json",
        type=str,
        help="Name of the JSON data file to use for evaluation (e.g., OCRBench_v2_new_5.json)",
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
    def __init__(self, json_data_file="OCRBench_v2_new.json"):
        super().__init__()
        # self.json_data = json.load(open('/home1/hxl/disk2/Backup/EAGLE/chenxn/OCRBench_v2/OCRBench_v2.json'))
        # self.json_data = json.load(open('/home1/hxl/disk2/Backup/EAGLE/chenxn/OCRBench_v2/OCRBench_v2_new.json'))
        json_data_path = os.path.join('/home1/hxl/disk2/Backup/EAGLE/chenxn/OCRBench_v2', json_data_file)
        self.json_data = json.load(open(json_data_path))
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


@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name
    )
    # print(f"image processor: {type(image_processor)}")
    model.eval()
    modality = 'image'
    test_dataset = VQADataset(json_data_file=args.json_data)
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn
    )

    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    outputs = []
    for i, data in enumerate(test_dataloader):
        data = data[0]
        image = Image.open(data.data_path).convert('RGB')
        image_tensor = process_images(
            images=[image],
            image_processor=image_processor,
            model_cfg=model.config
        )
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)
        if hasattr(image_tensor, "image_grid_thw"):
            image_grid_thw = image_tensor['image_grid_thw']
            image_tensor = image_tensor['pixel_values']
        else:
            image_grid_thw = None
            print("No image_grid_thw found, using None for image_grid_thw")
        question = data.question
        answer = data.answer

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
            image_grid_thw=image_grid_thw
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
        pbar.update(1)
    pbar.close()
    
    # Generate date-based output path
    current_date = datetime.now().strftime("%m%d")
    base_output_dir = '/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/ocrb/eagle_ocr'
    date_dir = os.path.join(base_output_dir, current_date)
    full_output_path = os.path.join(date_dir, args.output_path)
    
    # make sure the output directory exists
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    # with open(args.output_path, 'w') as output_file:
    #     json.dump(outputs, output_file)
    
    print("Save at", full_output_path)


def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

if __name__ == "__main__":
    args = parse_eval_args()
    evaluate(args=args)
