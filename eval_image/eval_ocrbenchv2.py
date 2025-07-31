import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import json
from datetime import datetime

import argparse
import logging
from typing import Union
from tqdm import tqdm
from PIL import Image

eval_logger = logging.getLogger("eval_eagle")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
        default=str(PROJECT_ROOT / "checkpoints/Images/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle"),
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
        help="The path to the output file where the result metrics will be saved.",
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
        default="OCRBench_v2_new_5.json",
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
        # self.json_data = json.load(open(str(PROJECT_ROOT / 'eval_image/OCRBench_v2/OCRBench_v2.json')))
        # self.json_data = json.load(open(str(PROJECT_ROOT / 'eval_image/OCRBench_v2/OCRBench_v2_new.json')))
        json_data_path = os.path.join(str(PROJECT_ROOT / 'eval_image/OCRBench_v2'), json_data_file)
        self.json_data = json.load(open(json_data_path))
        self.img_dir = str(PROJECT_ROOT / 'eval_image/OCRBench_v2')
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
def run_inference(args: Union[argparse.Namespace, None] = None) -> dict:
    """Run inference only and return raw predictions"""
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name
    )
    
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    
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
        outputs.append(
            {
                "question": question,
                "answer": answer,
                "prediction": text_outputs
            }
        )
        pbar.update(1)
    pbar.close()
    
    # Optionally save raw predictions as backup
    if args.output_path:
        # Generate date-based output path with model folder
        current_date = datetime.now().strftime("%m%d")
        base_output_dir = str(PROJECT_ROOT / 'eval_image/eagle_ocr')
        model_dir = os.path.join(base_output_dir, model_folder_name)
        date_dir = os.path.join(model_dir, current_date)
        full_output_path = os.path.join(date_dir, args.output_path)
        
        # make sure the output directory exists
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        with open(full_output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)
        
        print("Raw predictions saved at", full_output_path)
    
    print("Inference completed for OCRBench v2!")
    return {"ocrbenchv2": outputs}


def evaluate_predictions(inference_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Evaluate predictions using the OCRBench v2 evaluation pipeline"""
    # Note: If output_path already contains model directory, don't add it again
    # This prevents duplicate model directory names
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    
    # Get predictions from inference_results or load from file
    if inference_results is not None and "ocrbenchv2" in inference_results:
        outputs = inference_results["ocrbenchv2"]
        print("Processing OCRBench v2 predictions from inference results...")
    else:
        # Fallback to reading from file
        if args.output_path:
            # Check if output_path already contains model directory to avoid duplication
            if not args.output_path.endswith(model_folder_name):
                current_date = datetime.now().strftime("%m%d")
                base_output_dir = str(PROJECT_ROOT / 'eval_image/eagle_ocr')
                model_dir = os.path.join(base_output_dir, model_folder_name)
                date_dir = os.path.join(model_dir, current_date)
                full_output_path = os.path.join(date_dir, args.output_path)
            else:
                # Use output_path directly if it already contains model folder
                full_output_path = args.output_path
            
            if os.path.exists(full_output_path):
                with open(full_output_path, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                print("Processing OCRBench v2 predictions from file...")
            else:
                print(f"OCRBench v2 prediction file not found: {full_output_path}")
                return {}
        else:
            print("No inference results or output path provided")
            return {}
    
    try:
        # Import the get_result processing function
        from get_result import process_predictions
        
        # Process predictions using get_result.py logic
        data_names = ['all_bbox']  # Default data name for OCRBench v2
        predict_model = 'eagle'
        
        # Save predictions in the expected format first
        current_date = datetime.now().strftime("%m%d")
        base_output_dir = str(PROJECT_ROOT / 'eval_image/eagle_ocr')
        model_dir = os.path.join(base_output_dir, model_folder_name)
        eagle_ocr_dir = os.path.join(model_dir, current_date)
        predict_file = os.path.join(eagle_ocr_dir, 'all_bbox.json')
        
        os.makedirs(eagle_ocr_dir, exist_ok=True)
        with open(predict_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)
        
        # Process predictions
        processed_outputs = process_predictions(
            data_names=data_names,
            predict_model=predict_model,
            predict_path=predict_file,
            raw_data=args.json_data if args.json_data else 'OCRBench_v2.json'
        )
        
        # Save processed predictions for evaluation
        pred_folder_dir = str(PROJECT_ROOT / 'eval_image/MultimodalOCR-main/OCRBench_v2/pred_folder')
        os.makedirs(pred_folder_dir, exist_ok=True)
        processed_pred_file = os.path.join(pred_folder_dir, f'vqa_{predict_model}_{model_folder_name}.json')
        
        with open(processed_pred_file, "w", encoding="utf-8") as f:
            json.dump(processed_outputs, f, ensure_ascii=False, indent=4)
        
        print(f"Processed predictions saved to {processed_pred_file}")
        
        # Run OCRBench v2 evaluation script
        import subprocess
        import sys
        
        eval_script = str(PROJECT_ROOT / 'eval_image/MultimodalOCR-main/OCRBench_v2/eval_scripts/eval.py')
        eval_output_dir = str(PROJECT_ROOT / 'eval_image/eval_ocrbench')
        os.makedirs(eval_output_dir, exist_ok=True)
        eval_output_file = os.path.join(eval_output_dir, f'{predict_model}_{model_folder_name}.json')
        
        # Run evaluation
        eval_cmd = [
            sys.executable, eval_script,
            '--input_path', processed_pred_file,
            '--output_path', eval_output_file
        ]
        
        print(f"Running OCRBench v2 evaluation: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Evaluation script failed: {result.stderr}")
            return {
                "error": f"Evaluation script failed: {result.stderr}",
                "processed_predictions": len(processed_outputs),
                "total_predictions": len(outputs)
            }
        
        # Run score calculation
        score_script = str(PROJECT_ROOT / 'eval_image/MultimodalOCR-main/OCRBench_v2/eval_scripts/get_score.py')
        score_cmd = [
            sys.executable, score_script,
            '--json_file', eval_output_file
        ]
        
        print(f"Running OCRBench v2 score calculation: {' '.join(score_cmd)}")
        score_result = subprocess.run(score_cmd, capture_output=True, text=True)
        
        if score_result.returncode != 0:
            print(f"Score calculation failed: {score_result.stderr}")
        
        # Try to extract scores from the output
        score_output = score_result.stdout
        print("OCRBench v2 Evaluation Results:")
        print(score_output)
        
        # Parse scores using the dedicated function
        parsed_scores = parse_final_results(score_output)
        
        # Parse scores from output (this is a simplified version)
        evaluation_results = {
            "processed_predictions": len(processed_outputs),
            "total_predictions": len(outputs),
            "evaluation_output": score_output,
            "parsed_scores": parsed_scores,
            "eval_output_file": eval_output_file,
            "processed_pred_file": processed_pred_file
        }
        
        # Try to read the evaluation results file if it exists
        if os.path.exists(eval_output_file):
            with open(eval_output_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            evaluation_results["detailed_results"] = eval_data
            evaluation_results["total_evaluated"] = len(eval_data)
        
        return evaluation_results
        
    except ImportError as e:
        print(f"Warning: Could not import OCRBench v2 processing modules: {e}")
        print("Falling back to basic statistics...")
        return {
            "total_predictions": len(outputs),
            "error": f"Import error: {e}"
        }
    except Exception as e:
        print(f"Error during OCRBench v2 evaluation: {e}")
        return {
            "total_predictions": len(outputs),
            "error": str(e)
        }


def parse_output(evaluation_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Parse and summarize OCRBench v2 evaluation results"""
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    
    if evaluation_results is not None:
        # Use the passed evaluation results
        results = evaluation_results
        print("Parsing OCRBench v2 results from evaluation results...")
    else:
        # This is mainly for backward compatibility
        print("Warning: No evaluation results passed, this may indicate an incomplete evaluation flow")
        return {}
    
    # Extract summary information
    summary = {}
    if "total_predictions" in results:
        summary["total_predictions"] = results["total_predictions"]
    if "processed_predictions" in results:
        summary["processed_predictions"] = results["processed_predictions"]
    if "total_evaluated" in results:
        summary["total_evaluated"] = results["total_evaluated"]
    if "evaluation_output" in results:
        summary["has_evaluation_output"] = True
    if "parsed_scores" in results:
        summary["parsed_scores"] = results["parsed_scores"]
    if "error" in results:
        summary["error"] = results["error"]
    
    # Print summary
    print(f"OCRBench v2 Summary:")
    if "total_predictions" in summary:
        print(f"  - Total predictions: {summary['total_predictions']}")
    if "processed_predictions" in summary:
        print(f"  - Processed predictions: {summary['processed_predictions']}")
    if "total_evaluated" in summary:
        print(f"  - Total evaluated: {summary['total_evaluated']}")
    if "error" in summary:
        print(f"  - Error: {summary['error']}")
    
    # Print parsed scores if available
    if "parsed_scores" in summary and summary["parsed_scores"]:
        parsed_scores = summary["parsed_scores"]
        print(f"  - Parsed Evaluation Scores:")
        
        # Print English scores
        if "english_scores" in parsed_scores and parsed_scores["english_scores"]:
            print(f"    English Scores:")
            for task, score_info in parsed_scores["english_scores"].items():
                print(f"      {task}: {score_info['score']:.3f} (Count: {score_info['count']})")
        
        # Print Chinese scores
        if "chinese_scores" in parsed_scores and parsed_scores["chinese_scores"]:
            print(f"    Chinese Scores:")
            for task, score_info in parsed_scores["chinese_scores"].items():
                print(f"      {task}: {score_info['score']:.3f} (Count: {score_info['count']})")
        
        # Print overall scores
        if "overall_scores" in parsed_scores and parsed_scores["overall_scores"]:
            print(f"    Overall Scores:")
            if "english_overall" in parsed_scores["overall_scores"]:
                print(f"      English Overall: {parsed_scores['overall_scores']['english_overall']:.3f}")
            if "chinese_overall" in parsed_scores["overall_scores"]:
                print(f"      Chinese Overall: {parsed_scores['overall_scores']['chinese_overall']:.3f}")
        
        # Print detailed task statistics
        if "task_statistics" in parsed_scores and parsed_scores["task_statistics"]:
            task_stats = parsed_scores["task_statistics"]
            print(f"  - Task Statistics:")
            
            # English task analysis
            if task_stats.get("english_task_count", 0) > 0:
                print(f"    English Tasks Analysis:")
                print(f"      Total Tasks: {task_stats['english_task_count']}")
                print(f"      Total Test Items: {task_stats['total_english_count']}")
                print(f"      Weighted Average: {task_stats['english_weighted_average']:.3f}")
                
                if task_stats.get("english_best_task"):
                    best = task_stats["english_best_task"]
                    print(f"      Best Task: {best['task']} ({best['score']:.3f})")
                
                if task_stats.get("english_worst_task"):
                    worst = task_stats["english_worst_task"]
                    print(f"      Worst Task: {worst['task']} ({worst['score']:.3f})")
            
            # Chinese task analysis
            if task_stats.get("chinese_task_count", 0) > 0:
                print(f"    Chinese Tasks Analysis:")
                print(f"      Total Tasks: {task_stats['chinese_task_count']}")
                print(f"      Total Test Items: {task_stats['total_chinese_count']}")
                print(f"      Weighted Average: {task_stats['chinese_weighted_average']:.3f}")
                
                if task_stats.get("chinese_best_task"):
                    best = task_stats["chinese_best_task"]
                    print(f"      Best Task: {best['task']} ({best['score']:.3f})")
                
                if task_stats.get("chinese_worst_task"):
                    worst = task_stats["chinese_worst_task"]
                    print(f"      Worst Task: {worst['task']} ({worst['score']:.3f})")
            
            # Task breakdown by performance
            print(f"    Performance Breakdown:")
            all_tasks = []
            
            # Combine English and Chinese tasks
            for task in task_stats.get("english_tasks", []):
                all_tasks.append({"language": "English", **task})
            for task in task_stats.get("chinese_tasks", []):
                all_tasks.append({"language": "Chinese", **task})
            
            if all_tasks:
                # Sort by score
                all_tasks.sort(key=lambda x: x["score"], reverse=True)
                
                # Show top 3 and bottom 3 tasks
                print(f"      Top 3 Tasks:")
                for i, task in enumerate(all_tasks[:3]):
                    print(f"        {i+1}. {task['language']} - {task['task']}: {task['score']:.3f}")
                
                if len(all_tasks) > 3:
                    print(f"      Bottom 3 Tasks:")
                    for i, task in enumerate(all_tasks[-3:]):
                        rank = len(all_tasks) - 2 + i
                        print(f"        {rank}. {task['language']} - {task['task']}: {task['score']:.3f}")
    
    # Optionally save summary
    if args and args.output_path:
        eval_output_dir = str(PROJECT_ROOT / 'eval_image/eval_ocrbench')
        os.makedirs(eval_output_dir, exist_ok=True)
        summary_file = os.path.join(eval_output_dir, f'ocrbenchv2_summary_{model_folder_name}.json')
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "ocrbenchv2_summary": summary,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=4)
        print(f"OCRBench v2 summary saved to {summary_file}")
    
    return summary


def parse_final_results(score_output: str) -> dict:
    """
    Parse the final evaluation results from OCRBench v2 score output.
    
    Args:
        score_output: The stdout from the score calculation script
        
    Returns:
        A dictionary containing parsed scores for English, Chinese, and overall metrics
    """
    results = {
        "english_scores": {},
        "chinese_scores": {},
        "overall_scores": {},
        "task_statistics": {},
        "raw_output": score_output
    }
    
    if not score_output or not isinstance(score_output, str):
        return results
    
    lines = score_output.strip().split('\n')
    current_section = None
    
    # Initialize task statistics
    task_stats = {
        "english_tasks": [],
        "chinese_tasks": [],
        "total_english_count": 0,
        "total_chinese_count": 0,
        "english_weighted_score": 0.0,
        "chinese_weighted_score": 0.0
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identify sections
        if line == "English Scores:":
            current_section = "english"
            continue
        elif line == "Chinese Scores:":
            current_section = "chinese"
            continue
        elif line == "Overall Scores:":
            current_section = "overall"
            continue
        elif line == "End of Code!":
            break
            
        # Parse score lines
        if current_section == "english" and ":" in line and "Count:" in line:
            # Format: "task_name: score (Count: count)"
            try:
                parts = line.split(":")
                if len(parts) >= 2:
                    task_name = parts[0].strip()
                    score_part = parts[1].strip()
                    
                    # Extract score and count
                    if "(" in score_part and "Count:" in score_part:
                        score_str = score_part.split("(")[0].strip()
                        count_str = score_part.split("Count:")[1].strip().rstrip(")")
                        
                        score = float(score_str)
                        count = int(count_str)
                        
                        results["english_scores"][task_name] = {
                            "score": score,
                            "count": count
                        }
                        
                        # Update statistics
                        task_stats["english_tasks"].append({
                            "task": task_name,
                            "score": score,
                            "count": count
                        })
                        task_stats["total_english_count"] += count
                        task_stats["english_weighted_score"] += score * count
                        
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse English score line: {line}, error: {e}")
                
        elif current_section == "chinese" and ":" in line and "Count:" in line:
            # Format: "task_name: score (Count: count)"
            try:
                parts = line.split(":")
                if len(parts) >= 2:
                    task_name = parts[0].strip()
                    score_part = parts[1].strip()
                    
                    # Extract score and count
                    if "(" in score_part and "Count:" in score_part:
                        score_str = score_part.split("(")[0].strip()
                        count_str = score_part.split("Count:")[1].strip().rstrip(")")
                        
                        score = float(score_str)
                        count = int(count_str)
                        
                        results["chinese_scores"][task_name] = {
                            "score": score,
                            "count": count
                        }
                        
                        # Update statistics
                        task_stats["chinese_tasks"].append({
                            "task": task_name,
                            "score": score,
                            "count": count
                        })
                        task_stats["total_chinese_count"] += count
                        task_stats["chinese_weighted_score"] += score * count
                        
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse Chinese score line: {line}, error: {e}")
                
        elif current_section == "overall" and ":" in line:
            # Format: "English Overall Score: score" or "Chinese Overall Score: score"
            try:
                if line.startswith("English Overall Score:"):
                    score_str = line.split(":")[1].strip()
                    results["overall_scores"]["english_overall"] = float(score_str)
                elif line.startswith("Chinese Overall Score:"):
                    score_str = line.split(":")[1].strip()
                    results["overall_scores"]["chinese_overall"] = float(score_str)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse overall score line: {line}, error: {e}")
    
    # Calculate weighted averages and add task statistics
    if task_stats["total_english_count"] > 0:
        task_stats["english_weighted_average"] = task_stats["english_weighted_score"] / task_stats["total_english_count"]
    else:
        task_stats["english_weighted_average"] = 0.0
        
    if task_stats["total_chinese_count"] > 0:
        task_stats["chinese_weighted_average"] = task_stats["chinese_weighted_score"] / task_stats["total_chinese_count"]
    else:
        task_stats["chinese_weighted_average"] = 0.0
    
    # Sort tasks by score (descending)
    task_stats["english_tasks"].sort(key=lambda x: x["score"], reverse=True)
    task_stats["chinese_tasks"].sort(key=lambda x: x["score"], reverse=True)
    
    # Add detailed task analysis
    task_stats["english_best_task"] = task_stats["english_tasks"][0] if task_stats["english_tasks"] else None
    task_stats["english_worst_task"] = task_stats["english_tasks"][-1] if task_stats["english_tasks"] else None
    task_stats["chinese_best_task"] = task_stats["chinese_tasks"][0] if task_stats["chinese_tasks"] else None
    task_stats["chinese_worst_task"] = task_stats["chinese_tasks"][-1] if task_stats["chinese_tasks"] else None
    
    # Calculate task distribution
    task_stats["english_task_count"] = len(task_stats["english_tasks"])
    task_stats["chinese_task_count"] = len(task_stats["chinese_tasks"])
    
    results["task_statistics"] = task_stats
    
    return results


def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def evaluate_with_results(model_path, output_path=None):
    """
    Run OCRBench v2 evaluation and return results as a dictionary
    
    Args:
        model_path (str): Path to the model checkpoint
        output_path (str): Custom output path for results. If None, uses default.
    """
    import sys
    
    # Temporarily modify sys.argv to pass the model_path argument
    original_argv = sys.argv.copy()
    argv_list = ['eval_ocrbenchv2.py', '--model_path', model_path]
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
        
        # Return structured results
        return {
            "ocrbenchv2": results,
            "status": "completed"
        }
        
    except Exception as e:
        # raise e
        return {"error": f"Failed to evaluate OCRBench v2: {str(e)}"}

if __name__ == "__main__":
    args = parse_eval_args()
    
    # Support different execution modes using argparse
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
