import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('./')
sys.path.append('../')
import json
from pathlib import Path

import argparse
import logging
from typing import Union
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
from train_video1 import ModelArguments

eval_logger = logging.getLogger("eval_3dllm")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# try:
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle

# Import the video model loading function from eval_video_qwen
try:
    from eval_video_qwen import load_video_model
except ImportError:
    # If import fails, define a fallback function
    def load_video_model(args, modality='video'):
        raise ImportError("load_video_model not available, please check eval_video_qwen.py")

def is_qwen_model(model_path: str) -> bool:
    """
    Check if the model is a qwen-based model by examining the model path.
    
    Args:
        model_path: Path to the model
        
    Returns:
        bool: True if it's a qwen model, False otherwise
    """
    # Check if 'qwen' appears in the model path (case insensitive)
    model_path_lower = model_path.lower()
    return 'qwen' in model_path_lower or 'qwen2vl' in model_path_lower
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")

def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/Videos/merged_model/finetune-video-llama3.2-3b-merged1-qwen-0.98-0.02", 
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
        default='/home6/fzy/repos/EAGLE/eval_image/res_folder/images',
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


# video_base = str(Path("~/.cache/huggingface/activitynetqa/all_test").expanduser())
video_base = str(Path("~/.cache/huggingface/activitynetqa/all_test_8").expanduser())

def activitynetqa_doc_to_visual(doc):
    video_path = os.path.join(video_base, f"v_{doc['video_name']}.mp4")
    extensions = ["mp4", "webm", "mkv"]
    for ext in extensions:
        modified_path = video_path.replace("mp4", ext)
        if os.path.exists(modified_path):
            return [modified_path]
    return None
    sys.exit(f"video path:{video_path} does not exist, please check")
    
@torch.no_grad()
def evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path:
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    modality = 'video'
    
    # Check if this is a qwen model and use appropriate loading method
    if is_qwen_model(args.model_path):
        print("Detected qwen model, using load_video_model...")
        try:
            model, tokenizer, image_processor, modality = load_video_model(args, modality=modality)
            print(f"Successfully loaded qwen model using load_video_model")
            # For qwen models, use float16
            model_dtype = torch.float16
        except Exception as e:
            print(f"Failed to load qwen model with load_video_model: {e}")
            print("Falling back to traditional load_pretrained_model...")
            # Fall back to traditional method
            tokenizer, model, image_processor, max_length = load_pretrained_model(
                model_path=args.model_path,
                model_base=None,
                model_name=args.model_name
            )
            image_processor = image_processor.video_processor
            # For fallback, use float16 (as original code used)
            model_dtype = torch.float16
    else:
        print("Using traditional load_pretrained_model...")
        # Use traditional loading method for non-qwen models
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path=args.model_path,
            model_base=None,
            model_name=args.model_name
        )
        image_processor = image_processor.video_processor
        # For traditional models, use float16 (as original code used)
        model_dtype = torch.float16
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
        if video_file is None:
            print(f"Skipping {data['video_name']} due to missing video file")
            continue
        image_tensor = process_images(
            images=video_file,
            image_processor=image_processor,
            model_cfg=model.config,
            modality=modality,
        )
        image_tensor = image_tensor.to(dtype=model_dtype, device=args.device)
                # image = image_full['pixel_values_videos']
                # video_grid_thw = image_full['video_grid_thw']
        if hasattr(image_tensor, "video_grid_thw"):
            video_grid_thw = image_tensor['video_grid_thw']
            image_tensor = image_tensor['pixel_values_videos']
        else:
            video_grid_thw = None
            print("No video_grid_thw found, using None for video_grid_thw")
        question = data['question']
        answer = data['answer']

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
        pbar.update(1)
    pbar.close()
    
    # Create output directory if it doesn't exist
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, "acqa.json")
    else:
        output_file = "acqa.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    # with open(args.output_path, 'w') as output_file:
    #     json.dump(outputs, output_file)
    
    print("Save at", output_file)
    
    # Calculate and return results for integration with eval_video_all.py
    total_questions = len(outputs)
    valid_predictions = len([o for o in outputs if o['prediction'] and len(o['prediction']) > 0])
    
    return {
        'status': 'completed',
        'total_questions': total_questions,
        'valid_predictions': valid_predictions,
        'output_file': output_file,
        'sample_predictions': outputs[:5]  # Return first 5 samples for inspection
    }


def parse_output(evaluation_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Parse and summarize ActivityNetQA evaluation results"""
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path:
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    if evaluation_results is not None:
        # Use the passed evaluation results
        results = evaluation_results
        print("Parsing ActivityNetQA results from evaluation results...")
    else:
        # Load from file if no evaluation results passed
        if args.output_path:
            output_file = os.path.join(args.output_path, "acqa.json")
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                print(f"Loading ActivityNetQA results from {output_file}")
                
                # Calculate evaluation metrics from loaded data
                from word2number import w2n
                from num2words import num2words
                
                def equal(pred, gt):
                    if pred.lower() in gt.lower():
                        return True
                    if gt.lower() in pred.lower():
                        return True
                    # if gt is a number, convert to string and compare
                    try:
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
                
                scores = []
                for output in outputs:
                    question = output['question']
                    answer = output['answer']
                    prediction = output['prediction'][0] if output['prediction'] and len(output['prediction']) > 0 else ""
                    if equal(prediction, answer):
                        scores.append(1)
                    else:
                        scores.append(0)
                
                results = {
                    'total_questions': len(outputs),
                    'correct_answers': sum(scores),
                    'accuracy': sum(scores) / len(scores) * 100 if len(scores) > 0 else 0,
                    'valid_predictions': len([o for o in outputs if o['prediction'] and len(o['prediction']) > 0])
                }
            else:
                print(f"ActivityNetQA prediction file not found: {output_file}")
                return {}
        else:
            print("No evaluation results or output path provided")
            return {}
    
    # Extract summary information
    summary = {}
    if 'accuracy' in results:
        summary['accuracy'] = results['accuracy']
    if 'total_questions' in results:
        summary['total_questions'] = results['total_questions']
    if 'correct_answers' in results:
        summary['correct_answers'] = results['correct_answers']
    if 'valid_predictions' in results:
        summary['valid_predictions'] = results['valid_predictions']
    
    # Print summary
    if 'accuracy' in summary:
        print(f"ActivityNetQA Accuracy: {summary['accuracy']:.2f}%")
        if 'total_questions' in summary and 'correct_answers' in summary:
            print(f"  - Correct: {summary['correct_answers']}/{summary['total_questions']}")
    
    # Optionally save summary
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        summary_file = os.path.join(args.output_path, "acqa_evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "acqa_summary": summary,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=4)
        print(f"ActivityNetQA summary saved to {summary_file}")
    
    return summary


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
    

def evaluate_with_results(model_path, datasets=None, output_path=None):
    """
    Wrapper function for compatibility with eval_video_all.py
    
    Args:
        model_path: Path to the pretrained model
        datasets: List of dataset names to evaluate (not used for ACQA)
        output_path: Optional custom output directory. If None, uses unified structure.
    """
    import argparse
    
    # Create mock args object
    args = argparse.Namespace()
    args.model_path = model_path
    args.model_name = "eagle"
    args.device = "cuda"
    args.conv_template = "llama3"
    args.use_cache = None
    
    # Calculate unified output directory if no custom path provided
    if output_path is None:
        model_name = os.path.basename(model_path.rstrip('/'))
        base_output_dir = PROJECT_ROOT / "eval_video" / "res_folder" / "videos" / model_name
        args.output_path = str(base_output_dir)
    else:
        args.output_path = output_path
    
    # Call the main evaluate function
    return evaluate(args)


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
    # eval_res(args=args)