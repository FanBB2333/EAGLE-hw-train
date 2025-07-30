import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) 
import json
from datasets import load_dataset

import argparse
import logging
from typing import Union
from tqdm import tqdm
from PIL import Image

eval_logger = logging.getLogger("eval_image")

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
        default="/home6/fzy/repos/EAGLE/checkpoints/Images/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle",
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
        default='/home6/fzy/repos/EAGLE/eval_image/mme/eagle',
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

@torch.no_grad()
def run_inference(args: Union[argparse.Namespace, None] = None) -> dict:
    """Run inference only and return raw predictions"""
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name
    )
    # print(f"image processor: {type(image_processor)}")
    model.eval()
    modality = 'image'
    # test_dataset = VQADataset()
    test_dataset = load_dataset("lmms-lab/MME", split="test")
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=custom_collate_fn
    )

    pbar = tqdm(total=len(test_dataloader), desc="Model Responding")
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
            image_grid_thw=image_grid_thw
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs.append(
            {
                "question_id": data['question_id'],
                "question": question,
                "answer": answer,
                "prediction": text_outputs,
                "category": data['category'],
            }
        )
        pbar.update(1)
    pbar.close()
    
    # Optionally save raw predictions as backup
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, "mme.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)
        print("Raw predictions saved at", output_file)
    
    print("Inference completed for MME!")
    return {"mme": outputs}


def evaluate_predictions(inference_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Evaluate predictions by converting to MME format and calculating scores"""
    # Get predictions from inference_results or load from file
    if inference_results is not None and "mme" in inference_results:
        outputs = inference_results["mme"]
        print("Converting MME predictions from inference results...")
    else:
        # Fallback to reading from file
        if args.output_path:
            import os
            output_file = os.path.join(args.output_path, "mme.json")
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    outputs = json.load(f)
                print("Converting MME predictions from file...")
            else:
                print(f"MME prediction file not found: {output_file}")
                return {}
        else:
            print("No inference results or output path provided")
            return {}
    
    # Convert to MME format: save to tsv files for each category
    # Format: {category}.txt with lines: 000000012120.jpg	question	answer	prediction
    categories = set()
    for output in outputs:
        categories.add(output['category'])
    
    evaluation_results = {}
    
    # Create output directory for MME format files
    if args.output_path:
        mme_format_dir = os.path.join(args.output_path, "mme_format")
        os.makedirs(mme_format_dir, exist_ok=True)
        
        for category in categories:
            category_file = os.path.join(mme_format_dir, f"{category}.txt")
            with open(category_file, "w", encoding="utf-8") as f:
                for output in outputs:
                    if output['category'] == category:
                        question_id = output['question_id']
                        question_id = question_id.split('/')[-1]  # remove the file extension if exists
                        question = output['question'].replace("<image>","").strip()
                        answer = output['answer']
                        prediction = output['prediction'][0] if isinstance(output['prediction'], list) else output['prediction']
                        f.write(f"{question_id}\t{question}\t{answer}\t{prediction}\n")
        
        print(f"MME format files saved to {mme_format_dir}")
        
        # Calculate MME scores using the calculation script
        try:
            # Import the calculation functions using importlib
            import sys
            import os
            import importlib.util
            
            calculation_path = os.path.join(os.path.dirname(__file__), 'mme', 'calculation.py')
            if os.path.exists(calculation_path):
                spec = importlib.util.spec_from_file_location("calculation", calculation_path)
                calculation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(calculation_module)
                calculate_metrics = calculation_module.calculate_metrics
            else:
                raise ImportError(f"MME calculation script not found at {calculation_path}")
            
            calculator = calculate_metrics()
            eval_type_dict = {
                "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
                "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
            }
            
            category_scores = {}
            perception_total = 0
            cognition_total = 0
            
            for category in categories:
                category_file = os.path.join(mme_format_dir, f"{category}.txt")
                if os.path.exists(category_file):
                    # Read the category file
                    with open(category_file, "r") as f:
                        lines = f.readlines()
                    
                    # Process lines in chunks of 2 (as per MME format)
                    chunk_lines = list(calculator.divide_chunks(lines, 2))
                    
                    img_num = len(chunk_lines)
                    task_score = 0
                    acc_plus_correct_num = 0
                    gts = []
                    preds = []
                    
                    for img_items in chunk_lines:
                        if len(img_items) != 2:
                            continue  # Skip incomplete chunks
                        
                        img_correct_num = 0
                        for img_item in img_items:
                            parts = img_item.strip().split('\t')
                            if len(parts) < 4:
                                continue
                            
                            img_name, question, gt_ans, pred_ans = parts[:4]
                            gt_ans = gt_ans.lower().strip()
                            pred_ans = pred_ans.lower().strip()
                            
                            if gt_ans not in ["yes", "no"]:
                                continue
                            
                            parsed_pred = calculator.parse_pred_ans(pred_ans)
                            gts.append(gt_ans)
                            preds.append(parsed_pred)
                            
                            if gt_ans == parsed_pred:
                                img_correct_num += 1
                        
                        if img_correct_num == 2:
                            acc_plus_correct_num += 1
                    
                    if gts and preds and img_num > 0:
                        # Calculate metrics
                        metric_dict = calculator.compute_metric(gts, preds)
                        acc_plus = acc_plus_correct_num / img_num
                        metric_dict["acc_plus"] = acc_plus
                        
                        # Calculate task score (acc + acc_plus) * 100
                        for k, v in metric_dict.items():
                            if k in ["acc", "acc_plus"]:
                                task_score += v * 100
                        
                        category_scores[category] = {
                            "score": task_score,
                            "accuracy": metric_dict.get("acc", 0),
                            "accuracy_plus": acc_plus,
                            "count": len(gts)
                        }
                        
                        # Add to overall scores
                        for eval_type, eval_categories in eval_type_dict.items():
                            if category in eval_categories:
                                if eval_type == "Perception":
                                    perception_total += task_score
                                elif eval_type == "Cognition":
                                    cognition_total += task_score
                                break
            
            total_score = perception_total + cognition_total
            
            evaluation_results = {
                "category_scores": category_scores,
                "perception_score": perception_total,
                "cognition_score": cognition_total,
                "total_score": total_score,
                "total_categories": len(categories)
            }
            
            print(f"MME Evaluation Results:")
            print(f"Perception Score: {perception_total:.2f}")
            print(f"Cognition Score: {cognition_total:.2f}")
            print(f"Total Score: {total_score:.2f}")
            
        except ImportError as e:
            print(f"Warning: Could not import MME calculation module: {e}")
            print("Falling back to basic category counting...")
            evaluation_results = {
                "category_scores": {cat: {"count": len([o for o in outputs if o['category'] == cat])} for cat in categories},
                "total_categories": len(categories),
                "total_predictions": len(outputs)
            }
        except Exception as e:
            print(f"Error during MME evaluation: {e}")
            evaluation_results = {
                "category_scores": {cat: {"count": len([o for o in outputs if o['category'] == cat])} for cat in categories},
                "total_categories": len(categories),
                "total_predictions": len(outputs),
                "error": str(e)
            }
    
    return evaluation_results


def parse_output(evaluation_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Parse and summarize MME evaluation results"""
    if evaluation_results is not None:
        # Use the passed evaluation results
        results = evaluation_results
        print("Parsing MME results from evaluation results...")
    else:
        # This is mainly for backward compatibility - the new flow should pass evaluation_results
        print("Warning: No evaluation results passed, this may indicate an incomplete evaluation flow")
        return {}
    
    # Extract summary information
    summary = {}
    if "perception_score" in results:
        summary["perception_score"] = results["perception_score"]
    if "cognition_score" in results:
        summary["cognition_score"] = results["cognition_score"]
    if "total_score" in results:
        summary["total_score"] = results["total_score"]
    if "total_categories" in results:
        summary["total_categories"] = results["total_categories"]
    if "category_scores" in results:
        summary["category_count"] = len(results["category_scores"])
    
    # Print summary
    if "total_score" in summary:
        print(f"MME Total Score: {summary['total_score']:.2f}")
        if "perception_score" in summary and "cognition_score" in summary:
            print(f"  - Perception: {summary['perception_score']:.2f}")
            print(f"  - Cognition: {summary['cognition_score']:.2f}")
    
    # Optionally save summary
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        summary_file = os.path.join(args.output_path, "mme_evaluation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "mme_summary": summary,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=4)
        print(f"MME summary saved to {summary_file}")
    
    return summary
    

def pad_sequence(tokenizer, input_ids, batch_first, padding_value) -> torch.Tensor:
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids

def evaluate_with_results(model_path):
    """
    Run MME evaluation and return results as a dictionary
    """
    import sys
    
    # Temporarily modify sys.argv to pass only the model_path argument
    original_argv = sys.argv.copy()
    sys.argv = ['eval_mme.py', '--model_path', model_path]
    
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
            "mme": results,
            "status": "completed"
        }
        
    except Exception as e:
        raise e
        return {"error": f"Failed to evaluate MME: {str(e)}"}

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
