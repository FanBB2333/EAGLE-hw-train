import os
import multiprocessing
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add parent directory to path
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Parse arguments first to set environment variables early
def parse_args():
    parser = argparse.ArgumentParser(description="Run all image evaluation tasks")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/Images/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--sequential", 
        action="store_true",
        default=True,
        help="Run evaluations sequentially instead of in parallel"
    )
    parser.add_argument(
        "--datasets", 
        default="docvqa",
        help="Datasets to evaluate. Options: 'all', 'docvqa', 'mme', 'ocrbenchv2', or comma-separated list (e.g., 'docvqa,mme')"
    )
    parser.add_argument(
        "--gpus", 
        default="6",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Default: '0'"
    )
    return parser.parse_args()

# Parse arguments and set GPU environment early
args = parse_args()

# Preprocess model_path: convert to absolute path relative to PROJECT_ROOT if not already absolute
if not os.path.isabs(args.model_path):
    args.model_path = str(PROJECT_ROOT / args.model_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

def run_evaluation_internal(script_name, model_path):
    """Run evaluation internally and return results"""
    # try:
    # Import evaluation functions dynamically when needed
    if script_name == "eval_docvqa_textvqa_chartqa.py":
        from eval_docvqa_textvqa_chartqa import evaluate_with_results
    elif script_name == "eval_mme.py":
        from eval_mme import evaluate_with_results
    elif script_name == "eval_ocrbenchv2.py":
        from eval_ocrbenchv2 import evaluate_with_results
    else:
        return {
            'script': script_name,
            'status': 'error',
            'error': f'Unknown script: {script_name}'
        }
    
    # Call the evaluation function directly
    results = evaluate_with_results(model_path)
    print(f"✓ {script_name} completed successfully")
    return {
        'script': script_name,
        'status': 'success',
        'results': results
    }
        
    # except Exception as e:
    #     print(f"✗ Failed to run {script_name}: {str(e)}")
    #     raise e
    #     return {
    #         'script': script_name,
    #         'status': 'error',
    #         'error': str(e)
    #     }

def run_all(args):
    """Run all evaluation scripts"""
    # Define mapping between dataset names and script files
    dataset_scripts = {
        "docvqa": "eval_docvqa_textvqa_chartqa.py",
        "mme": "eval_mme.py", 
        "ocrbenchv2": "eval_ocrbenchv2.py",
    }
    
    eval_scripts = []
    
    # Parse datasets argument
    if args.datasets.lower() == "all":
        eval_scripts = list(dataset_scripts.values())
    else:
        # Parse comma-separated dataset names
        requested_datasets = [d.strip().lower() for d in args.datasets.split(",")]
        for dataset in requested_datasets:
            if dataset in dataset_scripts:
                eval_scripts.append(dataset_scripts[dataset])
            else:
                print(f"Warning: Unknown dataset '{dataset}'. Available options: {', '.join(dataset_scripts.keys())}")
    
    if not eval_scripts:
        print("No evaluation scripts to run")
        return
    
    print(f"Running {len(eval_scripts)} evaluation scripts...")
    print(f"Model path: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Sequential mode: {args.sequential}")
    print(f"CUDA_VISIBLE_DEVICES: {args.gpus}")
    print("-" * 50)
    
    results = []
    
    # Run evaluations internally and collect results
    if args.sequential:
        for script in eval_scripts:
            result = run_evaluation_internal(script, args.model_path)
            results.append(result)
    else:
        # Run evaluations in parallel using multiprocessing
        with multiprocessing.Pool() as pool:
            tasks = [(script, args.model_path) for script in eval_scripts]
            results = pool.starmap(run_evaluation_internal, tasks)
    
    # Print summary of results
    print("-" * 50)
    print("Evaluation Results Summary:")
    for result in results:
        print(f"Script: {result['script']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success' and 'results' in result:
            print(f"Results: {json.dumps(result['results'], indent=2)}")
        elif result['status'] == 'error':
            print(f"Error: {result['error']}")
        print("-" * 30)
    
    return results

if __name__ == "__main__":
    results = run_all(args)
    if results:
        print(f"\nReturned {len(results)} evaluation results")