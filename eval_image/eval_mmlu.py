import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set CUDA device visibility if needed
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  
from datasets import load_dataset

import argparse
import logging
import re
from typing import Union
from tqdm import tqdm

eval_logger = logging.getLogger("eval_mmlu")

# MMLU template format
MMLU_TEMPLATE = {
    "en": {
        "system": "The following are multiple choice questions (with answers) about {subject}.\n\n",
        "choice": "\n{choice}. {content}",
        "answer": "\nAnswer:",
    },
}

# try:
from eagle.model.builder import load_pretrained_model
from eagle.mm_utils import get_model_name_from_path, tokenizer_image_token
from eagle.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from eagle.conversation import conv_templates, SeparatorStyle
# except ImportError:
#     eval_logger.error("Please add a symbolic link pointing to the eagle folder of repo ")


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument(
        "--model_path", 
        default="/home6/fzy/repos/EAGLE/checkpoints/Images/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle",
        help="Pretrained path of model"
    )
    parser.add_argument(
        "--model_name", 
        default="eagle", 
        help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated list of MMLU subjects to evaluate. If None, evaluate all subjects.",
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


def format_mmlu_question(question, choices, subject):
    """Format MMLU question using the template"""
    template = MMLU_TEMPLATE["en"]
    
    # Add system prompt with subject
    formatted_question = template["system"].format(subject=subject.replace("_", " "))
    
    # Add the question
    formatted_question += question
    
    # Add choices
    choice_labels = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        if i < len(choice_labels):
            formatted_question += template["choice"].format(
                choice=choice_labels[i], 
                content=choice
            )
    
    # Add answer prompt
    formatted_question += template["answer"]
    
    return formatted_question

def evaluate_mmlu_answer(prediction, target_index):
    """Evaluate MMLU answer prediction against target"""
    prediction = prediction.strip().upper()
    choice_labels = ['A', 'B', 'C', 'D']
    
    # Get the target choice letter
    if 0 <= target_index < len(choice_labels):
        target_choice = choice_labels[target_index]
    else:
        return False
    
    # Check if prediction starts with the correct choice
    if prediction.startswith(target_choice):
        return True
    
    # Check if prediction contains only the correct choice letter
    if prediction == target_choice:
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
    modality = 'text'
    
    # Load MMLU dataset
    mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
    # {'question': 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.',
    # 'subject': 'abstract_algebra',
    # 'choices': ['0', '4', '2', '6'],
    # 'answer': 1}
    
    # Group by subject if specific subjects are requested
    if args.subjects:
        selected_subjects = [s.strip() for s in args.subjects.split(',')]
        mmlu_dataset = mmlu_dataset.filter(lambda x: x['subject'] in selected_subjects)
    
    # Group dataset by subject
    subjects = {}
    for item in mmlu_dataset:
        subject = item['subject']
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(item)
    
    print(f"Found {len(subjects)} subjects with {len(mmlu_dataset)} total questions")
    
    # Process each subject
    all_outputs = []
    overall_correct = 0
    overall_total = 0
    
    # Create overall progress bar
    total_questions = len(mmlu_dataset)
    pbar = tqdm(total=total_questions, desc="MMLU Evaluation Progress")
    
    for subject_name, subject_data in subjects.items():
        print(f"Running inference on {subject_name} ({len(subject_data)} questions)...")
        
        subject_outputs = []
        subject_correct = 0
        
        for i, data in enumerate(subject_data):
            question = data['question']
            choices = data['choices']
            answer_index = data['answer']
            subject = data['subject']
            
            # Format the question using MMLU template
            formatted_question = format_mmlu_question(question, choices, subject)
            
            # Prepare conversation
            conv = conv_templates[args.conv_template].copy()
            conv.append_message(conv.roles[0], formatted_question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Tokenize input
            input_ids = tokenizer(
                prompt_question,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length if max_length else 2048
            ).input_ids.to(args.device)
            
            attention_mask = torch.ones_like(input_ids).to(args.device)

            gen_kwargs = {}
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 10  # Short answer for multiple choice
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            
            try:
                with torch.no_grad():
                    cont = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=args.use_cache,
                        modality=modality,
                    )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                
                # Extract only the generated part (remove the input prompt)
                prediction = text_outputs[0]
                if prompt_question in prediction:
                    prediction = prediction.replace(prompt_question, "").strip()
                
                # Evaluate the answer
                is_correct = evaluate_mmlu_answer(prediction, answer_index)
                if is_correct:
                    subject_correct += 1
                    overall_correct += 1
                
                subject_outputs.append({
                    "question_id": f"{subject_name}_{i}",
                    "subject": subject_name,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer_choice": ['A', 'B', 'C', 'D'][answer_index] if 0 <= answer_index < 4 else "Unknown",
                    "prediction": prediction,
                    "correct": is_correct
                })
                
            except Exception as e:
                print(f"Error processing sample {i} in {subject_name}: {e}")
                subject_outputs.append({
                    "question_id": f"{subject_name}_{i}",
                    "subject": subject_name,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer_choice": ['A', 'B', 'C', 'D'][answer_index] if 0 <= answer_index < 4 else "Unknown",
                    "prediction": "ERROR",
                    "correct": False
                })
            
            pbar.update(1)
        
        # Calculate subject accuracy
        subject_accuracy = subject_correct / len(subject_data) if len(subject_data) > 0 else 0
        print(f"{subject_name} Accuracy: {subject_accuracy:.4f} ({subject_correct}/{len(subject_data)})")
        
        # Add to overall outputs
        all_outputs.extend(subject_outputs)
        overall_total += len(subject_data)
    
    # Close the overall progress bar
    pbar.close()
    
    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    print(f"Overall MMLU Accuracy: {overall_accuracy:.4f} ({overall_correct}/{overall_total})")
    
    # Prepare results dictionary
    results = {
        "overall_accuracy": overall_accuracy,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "subject_count": len(subjects),
        "all_predictions": all_outputs,
        "subject_results": {}
    }
    
    # Add subject-specific results
    for subject_name, subject_data in subjects.items():
        subject_outputs = [output for output in all_outputs if output["subject"] == subject_name]
        subject_correct = sum(1 for output in subject_outputs if output["correct"])
        subject_accuracy = subject_correct / len(subject_data) if len(subject_data) > 0 else 0
        
        results["subject_results"][subject_name] = {
            "accuracy": subject_accuracy,
            "correct": subject_correct,
            "total": len(subject_data),
            "predictions": subject_outputs
        }
    
    # Optionally save results to files for backup
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        overall_file = os.path.join(args.output_path, "mmlu_overall_results.json")
        with open(overall_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # Save subject-specific results
        for subject_name, subject_result in results["subject_results"].items():
            subject_file = os.path.join(args.output_path, f"{subject_name}_results.json")
            with open(subject_file, "w", encoding="utf-8") as f:
                json.dump({
                    "subject": subject_name,
                    **subject_result
                }, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to {args.output_path}")
    
    print("Inference completed for MMLU!")
    return results


def evaluate_predictions(inference_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Evaluate predictions and calculate metrics"""
    # Note: If output_path already contains model directory, don't add it again
    # This prevents duplicate model directory names
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path and not args.output_path.endswith(model_folder_name):
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    if inference_results is not None:
        # Use the passed inference results
        results = inference_results
        print(f"Evaluating MMLU results from inference...")
    else:
        # Fallback to reading from file (for backward compatibility)
        overall_file = os.path.join(args.output_path, "mmlu_overall_results.json")
        
        if not os.path.exists(overall_file):
            print(f"Overall results file not found: {overall_file}")
            return {}
            
        print(f"Loading existing MMLU results...")
        
        # Load overall results
        with open(overall_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    print(f"Overall MMLU Accuracy: {results['overall_accuracy']:.4f} ({results['overall_correct']}/{results['overall_total']})")
    print(f"Evaluated {results['subject_count']} subjects")
    
    # Print subject-wise results
    if "subject_results" in results:
        for subject_name, subject_result in results["subject_results"].items():
            print(f"{subject_name}: {subject_result['accuracy']:.4f} ({subject_result['correct']}/{subject_result['total']})")
    else:
        # Fallback for old format or file-based results
        if args and args.output_path and os.path.exists(args.output_path):
            for subject_file in os.listdir(args.output_path):
                if subject_file.endswith("_results.json") and subject_file != "mmlu_overall_results.json":
                    subject_path = os.path.join(args.output_path, subject_file)
                    with open(subject_path, "r", encoding="utf-8") as f:
                        subject_results = json.load(f)
                    print(f"{subject_results['subject']}: {subject_results['accuracy']:.4f} ({subject_results['correct']}/{subject_results['total']})")
    
    return results


def parse_output(evaluation_results: dict = None, args: Union[argparse.Namespace, None] = None) -> dict:
    """Parse and summarize results from MMLU evaluation"""
    # Extract model name from model_path for subfolder creation
    model_folder_name = os.path.basename(args.model_path.rstrip('/'))
    if args.output_path:
        args.output_path = os.path.join(args.output_path, model_folder_name)
    
    if evaluation_results is not None:
        # Use the passed evaluation results
        results = evaluation_results
    else:
        # Fallback to reading from file (for backward compatibility)
        output_path = args.output_path
        overall_file = os.path.join(output_path, "mmlu_overall_results.json")
        
        if not os.path.exists(overall_file):
            print(f"Overall results file not found: {overall_file}")
            return {}
        
        with open(overall_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    
    summary = {
        "overall_accuracy": results["overall_accuracy"],
        "overall_correct": results["overall_correct"],
        "overall_total": results["overall_total"],
        "subject_count": results["subject_count"]
    }
    
    print(f"MMLU Overall: {results['overall_accuracy']:.4f} ({results['overall_correct']}/{results['overall_total']})")
    print(f"Subjects evaluated: {results['subject_count']}")
    
    # Save summary if output path is provided
    if args and args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        summary_file = os.path.join(args.output_path, "mmlu_evaluation_summary.json")
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

def evaluate_with_results(model_path, output_path=None):
    """
    Run MMLU evaluation and return results as a dictionary
    
    Args:
        model_path (str): Path to the model checkpoint
        output_path (str): Custom output path for results. If None, uses default.
    """
    import sys
    
    # Temporarily modify sys.argv to pass the model_path argument
    original_argv = sys.argv.copy()
    argv_list = ['eval_mmlu.py', '--model_path', model_path]
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
        summary = parse_output(evaluation_results=evaluation_results, args=args)
        
        # Return structured results
        return {
            "mmlu": summary,
            "status": "completed"
        }
        
    except Exception as e:
        raise e
        return {"error": f"Failed to evaluate MMLU: {str(e)}"}

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
