import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
import json

import argparse
import logging
from typing import Union
from tqdm import tqdm
from jiwer import wer

eval_logger = logging.getLogger("eval_3d")

from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle

from eval.dataset.conversation import ConversationDataset
# from eval.utils import DEFAULT_POINT_TOKEN

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/final_result1/3d_finetune_1epoch", 
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
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name
    )
    model.eval()
    modality = 'image'
    if 'video' in args.model_path.lower() or '3d' in args.model_path.lower():
        modality = 'video'
    elif 'audio' in args.model_path.lower():
        modality = 'audio'
    test_dataset = ConversationDataset(
        data_path="/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Audio",
        ann_path="/home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Audio/LibriSpeech/dev-clean.json"
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn
    )

    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
    outputs = []
    preds, gts = [], []
    for i, data in enumerate(test_dataloader):
        audio_id, audio_file, question, gt = data[0]
 
        image_tensor = image_processor(audio_file, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.to(dtype=torch.float16, device=args.device)

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
        response = text_outputs[0].strip().upper()
        outputs.append({
            'audio_id': audio_id,
            'question': question,
            'response': response,
            'gt': gt,
        })
        preds.append(response)
        gts.append(gt)
        pbar.update(1)
    pbar.close()

    with open(args.output_path, 'w') as output_file:
        json.dump(outputs, output_file)
    print(f"WER: {wer(gts, preds):.2f}%")


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
