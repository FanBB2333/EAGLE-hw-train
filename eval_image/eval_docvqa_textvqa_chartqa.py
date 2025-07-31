import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from accelerate import PartialState
import sys
sys.path.append('./')
import json
from datasets import load_dataset

import argparse
import logging
import re
from typing import Union
from tqdm import tqdm
from PIL import Image

eval_logger = logging.getLogger("eval_image")

# try:
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle/checkpoint-30000", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle", 
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-epoch2",
        # default="/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc",
        default="./checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle-inc3",
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
        "--datasets",
        # default="textvqa",
        default="textvqa,docvqa,chartqa",
        help="Comma-separated list of datasets to evaluate. Options: textvqa,docvqa,chartqa. Default: textvqa",
    )
    parser.add_argument(
        "--only_inference",
        action="store_true",
        help="Only run inference and save predictions without evaluation",
    )
    parser.add_argument(
        "--only_eval",
        action="store_true", 
        help="Only evaluate existing predictions without running inference",
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

def exact_match(prediction, target):
    """Exact string match after normalization"""
    pred_normalized = prediction.strip().lower()
    target_normalized = target.strip().lower()
    return pred_normalized == target_normalized

def number_match(prediction, target):
    """Extract and match numbers from prediction and target"""
    import re
    
    def extract_numbers(text):
        # Extract all numbers (including decimals) from text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(num) for num in numbers if num]
    
    pred_numbers = extract_numbers(prediction)
    target_numbers = extract_numbers(target)
    
    if not pred_numbers or not target_numbers:
        return False
    
    # Check if any predicted number matches any target number
    for pred_num in pred_numbers:
        for target_num in target_numbers:
            if abs(pred_num - target_num) < 1e-6:  # Allow small floating point errors
                return True
    return False

def evaluate_answers(prediction, targets):
    """Evaluate prediction against multiple possible answers"""
    if not targets:
        return False
    
    prediction = prediction.strip()
    
    # Try exact match first
    for target in targets:
        if exact_match(prediction, target):
            return True
    
    # Try number match
    for target in targets:
        if number_match(prediction, target):
            return True
    
    return False


@torch.no_grad()
def run_inference(args: Union[argparse.Namespace, None] = None) -> dict:
    """Run inference only and return raw predictions"""
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name
    )
    
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path:
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    model.eval()
    modality = 'image'
    
    # Parse datasets from args
    selected_datasets = [ds.strip() for ds in args.datasets.split(',')]
    
    # Load test datasets based on selected datasets
    available_datasets = {
        'textvqa': lambda: load_dataset("lmms-lab/textvqa", split="validation"),
        'docvqa': lambda: load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation"),
        'chartqa': lambda: load_dataset("lmms-lab/ChartQA", split="test"),
    }
    
    test_datasets = {}
    for dataset_name in selected_datasets:
        if dataset_name in available_datasets:
            print(f"Loading {dataset_name} dataset...")
            test_datasets[dataset_name] = available_datasets[dataset_name]()
        else:
            print(f"Warning: Unknown dataset '{dataset_name}'. Available datasets: {list(available_datasets.keys())}")
    
    if not test_datasets:
        print("No valid datasets selected. Exiting.")
        return {}
    
    # Collect all results
    all_results = {}
    
    # Process each dataset
    for dataset_name, test_dataset in test_datasets.items():
        print(f"Running inference on {dataset_name}...")
        
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=custom_collate_fn,
            batch_size=1
        )

        pbar = tqdm(total=len(test_dataloader), desc=f"Model Inference on {dataset_name}")
        outputs = []
        
        for i, data in enumerate(test_dataloader):
            data = data[0]
            image = data['image']
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
            
            question = data['question']
            
            # Handle different answer formats for different datasets
            if dataset_name == 'textvqa':
                # TextVQA answers is a list of strings, filter out empty ones
                answers = [answer.strip() for answer in data['answers'] if answer.strip()]
                question_id = str(data['question_id'])
            elif dataset_name == 'docvqa':
                # DocVQA answers is a list of strings or None
                if data['answers'] is not None and isinstance(data['answers'], list):
                    answers = [answer.strip() for answer in data['answers'] if answer.strip()]
                else:
                    answers = []
                question_id = str(data['questionId'])
            elif dataset_name == 'chartqa':
                # ChartQA has 'answer' field (single string)
                answers = [data['answer'].strip()] if data['answer'] else []
                question_id = f"{dataset_name}_{i}"
            else:
                answers = []
                question_id = f"{dataset_name}_{i}"

            # Add image token if not present
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
            
            input_ids = pad_sequence(
                tokenizer=tokenizer,
                input_ids=[input_ids], 
                batch_first=True, 
                padding_value=pad_token_ids
            ).to(args.device)
            attention_masks = input_ids.ne(pad_token_ids).to(args.device)

            gen_kwargs = {}
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 50
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            
            try:
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
                
                # Extract only the generated part (remove the input prompt)
                prediction = text_outputs[0]
                if prompt_question in prediction:
                    prediction = prediction.replace(prompt_question, "").strip()
                
                outputs.append({
                    "question_id": question_id,
                    "question": question,
                    "answers": answers,
                    "prediction": prediction,
                    "dataset": dataset_name
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                outputs.append({
                    "question_id": question_id,
                    "question": question,
                    "answers": answers,
                    "prediction": "ERROR",
                    "dataset": dataset_name
                })
            
            pbar.update(1)
        
        pbar.close()
        
        # Store results for this dataset
        all_results[dataset_name] = {
            "dataset": dataset_name,
            "total": len(outputs),
            "predictions": outputs
        }
        
        # Optionally save raw predictions for this dataset as backup
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            output_file = os.path.join(args.output_path, f"{dataset_name}_predictions.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results[dataset_name], f, ensure_ascii=False, indent=4)
            
            print(f"Predictions saved to {output_file}")
    
    print("Inference completed for all datasets!")
    return all_results


def evaluate_predictions(inference_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Evaluate predictions and calculate metrics"""
    # Note: If output_path already contains model directory, don't add it again
    # This prevents duplicate model directory names
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path and not args.output_path.endswith(model_folder_name):
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    # Parse datasets from args
    selected_datasets = [ds.strip() for ds in args.datasets.split(',')]
    all_evaluation_results = {}
    
    for dataset_name in selected_datasets:
        # Use inference_results if provided, otherwise load from file
        if inference_results is not None and dataset_name in inference_results:
            # Use the passed inference results
            data = inference_results[dataset_name]
            print(f"Evaluating predictions for {dataset_name} from inference results...")
        else:
            # Fallback to reading from file
            prediction_file = os.path.join(args.output_path, f"{dataset_name}_predictions.json")
            
            if not os.path.exists(prediction_file):
                print(f"Prediction file not found for {dataset_name}: {prediction_file}")
                continue
                
            print(f"Evaluating predictions for {dataset_name} from file...")
            
            # Load predictions
            with open(prediction_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        predictions = data["predictions"]
        correct_count = 0
        total_count = len(predictions)
        
        # Evaluate each prediction
        evaluated_results = []
        for pred in predictions:
            # Skip evaluation if no ground truth answers available
            if not pred["answers"]:
                evaluated_results.append({
                    **pred,
                    "correct": None  # Mark as unevaluable
                })
                continue
                
            is_correct = evaluate_answers(pred["prediction"], pred["answers"])
            if is_correct:
                correct_count += 1
            
            evaluated_results.append({
                **pred,
                "correct": is_correct
            })
        
        # Calculate accuracy (only count samples with ground truth)
        evaluable_count = sum(1 for result in evaluated_results if result["correct"] is not None)
        accuracy = correct_count / evaluable_count if evaluable_count > 0 else 0
        print(f"{dataset_name} Accuracy: {accuracy:.4f} ({correct_count}/{evaluable_count}) [Total samples: {total_count}]")
        
        # Store evaluation results
        all_evaluation_results[dataset_name] = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "correct": correct_count,
            "evaluable": evaluable_count,
            "total": total_count,
            "results": evaluated_results
        }
        
        # Optionally save evaluation results as backup
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
            result_file = os.path.join(args.output_path, f"{dataset_name}_results.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(all_evaluation_results[dataset_name], f, ensure_ascii=False, indent=4)
            
            print(f"Evaluation results saved to {result_file}")
    
    print("Evaluation completed for all datasets!")
    return all_evaluation_results


def parse_output(evaluation_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Parse and summarize results from all datasets"""
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path:
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    # Parse datasets from args
    selected_datasets = [ds.strip() for ds in args.datasets.split(',')]
    summary = {}
    
    for dataset_name in selected_datasets:
        # Use evaluation_results if provided, otherwise load from file
        if evaluation_results is not None and dataset_name in evaluation_results:
            # Use the passed evaluation results
            results = evaluation_results[dataset_name]
            print(f"Parsing results for {dataset_name} from evaluation results...")
        else:
            # Fallback to reading from file
            if args.output_path:
                result_file = os.path.join(args.output_path, f"{dataset_name}_results.json")
                if os.path.exists(result_file):
                    with open(result_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    print(f"Parsing results for {dataset_name} from file...")
                else:
                    print(f"Result file not found for {dataset_name}: {result_file}")
                    continue
            else:
                print(f"No evaluation results or output path provided for {dataset_name}")
                continue
        
        # Handle both old and new result format
        evaluable = results.get("evaluable", results.get("total", 0))
        
        summary[dataset_name] = {
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "evaluable": evaluable,
            "total": results["total"]
        }
        
        print(f"{dataset_name}: {results['accuracy']:.4f} ({results['correct']}/{evaluable}) [Total: {results['total']}]")
    
    # Optionally save summary
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        summary_file = os.path.join(args.output_path, "evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        
        print(f"Summary saved to {summary_file}")
    
    return summary
    

def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def evaluate_with_results(model_path, datasets="textvqa,docvqa,chartqa", output_path=None):
    """
    Run evaluation and return results as a dictionary
    
    Args:
        model_path (str): Path to the model checkpoint
        datasets (str): Comma-separated list of datasets to evaluate. 
                       Options: textvqa,docvqa,chartqa. Default: "textvqa,docvqa,chartqa"
        output_path (str): Custom output path for results. If None, uses default.
    """
    import sys
    
    # Temporarily modify sys.argv to pass the model_path and datasets arguments
    original_argv = sys.argv.copy()
    argv_list = ['eval_docvqa_textvqa_chartqa.py', '--model_path', model_path, '--datasets', datasets]
    if output_path:
        argv_list.extend(['--output_path', output_path])
    sys.argv = argv_list
    
    try:
        # Use the existing parse_eval_args function to get default parameters
        args = parse_eval_args()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    try:
        # Run the complete evaluation pipeline with data passing
        inference_results = run_inference(args=args)
        evaluation_results = evaluate_predictions(inference_results=inference_results, args=args)
        results = parse_output(evaluation_results=evaluation_results, args=args)
        
        # Return only the results for datasets that were actually evaluated
        structured_results = {}
        for dataset_name in results.keys():
            structured_results[dataset_name] = results[dataset_name]
        
        structured_results["status"] = "completed"
        return structured_results
        
    except Exception as e:
        raise e
        return {"error": f"Failed to evaluate DocVQA/TextVQA/ChartQA: {str(e)}"}

if __name__ == "__main__":
    args = parse_eval_args()
    
    if args.only_inference:
        # Only run inference and save predictions
        inference_results = run_inference(args=args)
    elif args.only_eval:
        # Only evaluate existing predictions
        evaluation_results = evaluate_predictions(args=args)
        summary = parse_output(evaluation_results=evaluation_results, args=args)
    else:
        # Run both inference and evaluation (default behavior)
        inference_results = run_inference(args=args)
        evaluation_results = evaluate_predictions(inference_results=inference_results, args=args)
        summary = parse_output(evaluation_results=evaluation_results, args=args)
