import multiprocessing
import argparse
import subprocess
import sys
import os
from pathlib import Path

# Parse arguments first to set environment variables early
def parse_args():
    parser = argparse.ArgumentParser(description="Run all image evaluation tasks")
    parser.add_argument(
        "--model_path", 
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--sequential", 
        action="store_true",
        help="Run evaluations sequentially instead of in parallel"
    )
    parser.add_argument(
        "--datasets", 
        default="all",
        help="Datasets to evaluate. Options: 'all', 'docvqa', 'mme', 'ocrbenchv2', or comma-separated list (e.g., 'docvqa,mme')"
    )
    parser.add_argument(
        "--gpus", 
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Default: '0'"
    )
    return parser.parse_args()

# Parse arguments and set GPU environment early
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

CURRENT_PATH = Path(__file__).parent

def run_evaluation(script_name, model_path):
    """Run a single evaluation script"""
    script_path = CURRENT_PATH / script_name
    cmd = [
        sys.executable, 
        str(script_path), 
        "--model_path", model_path
    ]
    
    # Prepare environment variables for subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Environment: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"✗ Failed to run {script_name}: {str(e)}")

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
    
    if args.sequential:
        # Run evaluations sequentially
        for script in eval_scripts:
            run_evaluation(script, args.model_path)
    else:
        # Run evaluations in parallel using multiprocessing
        with multiprocessing.Pool() as pool:
            tasks = [(script, args.model_path) for script in eval_scripts]
            pool.starmap(run_evaluation, tasks)
    
    print("-" * 50)
    print("All evaluations completed!")

if __name__ == "__main__":
    run_all(args)