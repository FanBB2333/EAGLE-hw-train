"""
Comprehensive Image Evaluation Script

This script runs multiple image evaluation tasks and saves results to organized files.

Usage Examples:
    # Run MMLU evaluation only
    python eval_image_all.py --datasets mmlu
    
    # Run multiple evaluations
    python eval_image_all.py --datasets mmlu,mme,docvqa
    
    # Run only TextVQA from the docvqa script
    python eval_image_all.py --datasets textvqa
    
    # Run DocVQA and ChartQA only
    python eval_image_all.py --datasets docvqa,chartqa
    
    # Run all evaluations with custom model
    python eval_image_all.py --datasets all --model_path /path/to/model
    
    # Use custom output directory
    python eval_image_all.py --datasets mmlu --output_dir ./my_results

Features:
    - Automatic result saving with timestamps and metadata
    - Support for sequential and parallel execution
    - Detailed result formatting and summary statistics
    - Integration with multiple evaluation scripts
    - Fine-grained dataset control for docvqa/textvqa/chartqa evaluations
"""

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
        default="all",
        help="Datasets to evaluate. Options: 'all', 'docvqa', 'textvqa', 'chartqa', 'mme', 'ocrbenchv2', 'mmlu', or comma-separated list (e.g., 'textvqa,docvqa'). For docvqa/textvqa/chartqa, you can specify individual datasets or combinations."
    )
    parser.add_argument(
        "--gpus", 
        default="4",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). Default: '0'"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Custom output directory for saving results. If not specified, uses default location."
    )
    return parser.parse_args()

# Parse arguments and set GPU environment early
args = parse_args()

# Preprocess model_path: convert to absolute path relative to PROJECT_ROOT if not already absolute
if not os.path.isabs(args.model_path):
    args.model_path = str(PROJECT_ROOT / args.model_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

def run_evaluation_internal(script_name, model_path, datasets=None):
    """Run evaluation internally and return results"""
    try:
        # Import evaluation functions dynamically when needed
        if script_name == "eval_docvqa_textvqa_chartqa.py":
            from eval_docvqa_textvqa_chartqa import evaluate_with_results
        elif script_name == "eval_mme.py":
            from eval_mme import evaluate_with_results
        elif script_name == "eval_ocrbenchv2.py":
            from eval_ocrbenchv2 import evaluate_with_results
        elif script_name == "eval_mmlu.py":
            from eval_mmlu import evaluate_with_results
        else:
            return {
                'script': script_name,
                'status': 'error',
                'error': f'Unknown script: {script_name}'
            }
        
        # Call the evaluation function directly
        print(f"Starting evaluation for {script_name}...")
        
        # Create dataset-specific output directory for this evaluation
        model_name = os.path.basename(model_path.rstrip('/'))
        base_output_dir = PROJECT_ROOT / "eval_image" / "res_folder" / "images" / model_name
        
        # Determine dataset name for directory creation
        if script_name == "eval_docvqa_textvqa_chartqa.py":
            if datasets:
                dataset_dir_name = datasets.replace(',', '_')
            else:
                dataset_dir_name = "textvqa_docvqa_chartqa"
        elif script_name == "eval_mme.py":
            dataset_dir_name = "mme"
        elif script_name == "eval_ocrbenchv2.py":
            dataset_dir_name = "ocrbenchv2"
        elif script_name == "eval_mmlu.py":
            dataset_dir_name = "mmlu"
        else:
            dataset_dir_name = "unknown"
        
        dataset_output_dir = base_output_dir / dataset_dir_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # For docvqa_textvqa_chartqa script, pass datasets parameter if provided
        if script_name == "eval_docvqa_textvqa_chartqa.py" and datasets:
            results = evaluate_with_results(model_path, datasets, str(dataset_output_dir))
        elif script_name == "eval_docvqa_textvqa_chartqa.py":
            results = evaluate_with_results(model_path, "textvqa,docvqa,chartqa", str(dataset_output_dir))
        elif script_name == "eval_mme.py":
            results = evaluate_with_results(model_path, str(dataset_output_dir))
        elif script_name == "eval_mmlu.py":
            results = evaluate_with_results(model_path, str(dataset_output_dir))
        elif script_name == "eval_ocrbenchv2.py":
            results = evaluate_with_results(model_path, str(dataset_output_dir))
        else:
            # Fallback for unknown scripts
            results = evaluate_with_results(model_path)
        
        print(f"✓ {script_name} completed successfully")
        
        # Handle different return formats
        if isinstance(results, dict):
            if 'status' in results and results['status'] == 'completed':
                # New format with status field
                return {
                    'script': script_name,
                    'status': 'success',
                    'results': results,
                    'dataset_output_dir': str(dataset_output_dir)
                }
            elif 'error' in results:
                # Error case
                return {
                    'script': script_name,
                    'status': 'error',
                    'error': results['error']
                }
            else:
                # Legacy format or direct results
                return {
                    'script': script_name,
                    'status': 'success',
                    'results': results,
                    'dataset_output_dir': str(dataset_output_dir)
                }
        else:
            # Unexpected format
            return {
                'script': script_name,
                'status': 'success',
                'results': {'raw_output': results},
                'dataset_output_dir': str(dataset_output_dir)
            }
            
    except Exception as e:
        print(f"✗ Failed to run {script_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'script': script_name,
            'status': 'error',
            'error': str(e)
        }

def save_results_to_file(results, args):
    """Save evaluation results to a JSON file with metadata"""
    import datetime
    
    # Create base output directory and model-specific directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = PROJECT_ROOT / "eval_image" / "res_folder" / "images"
    
    # Extract model name for directory structure
    model_name = os.path.basename(args.model_path.rstrip('/'))
    model_output_dir = base_output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    datasets_str = args.datasets.replace(",", "_")
    if len(datasets_str) > 20:
        datasets_str = datasets_str[:17] + "..."
    
    # Create summary filename in model directory
    summary_filename = f"summary_{datasets_str}_{timestamp}.json"
    output_file = model_output_dir / summary_filename
    
    # Prepare metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": args.model_path,
        "model_name": model_name,
        "datasets_evaluated": args.datasets,
        "sequential_mode": args.sequential,
        "gpu_devices": args.gpus,
        "total_evaluations": len(results),
        "successful_evaluations": len([r for r in results if r['status'] == 'success']),
        "failed_evaluations": len([r for r in results if r['status'] == 'error'])
    }
    
    # Prepare summary statistics
    summary_stats = {}
    for result in results:
        if result['status'] == 'success' and 'results' in result:
            script_name = result['script'].replace('.py', '')
            
            if script_name == 'eval_mmlu' and 'mmlu' in result['results']:
                # Extract MMLU specific stats
                mmlu_data = result['results']['mmlu']
                summary_stats[script_name] = {
                    "overall_accuracy": mmlu_data.get('overall_accuracy'),
                    "correct_answers": mmlu_data.get('overall_correct'),
                    "total_questions": mmlu_data.get('overall_total'),
                    "subjects_count": mmlu_data.get('subject_count')
                }
            elif isinstance(result['results'], dict):
                # Extract general stats for other evaluations
                stats = {}
                for key, value in result['results'].items():
                    if isinstance(value, (int, float)):
                        stats[key] = value
                    elif isinstance(value, dict) and 'accuracy' in str(value).lower():
                        # Try to find accuracy metrics
                        for sub_key, sub_value in value.items():
                            if 'accuracy' in sub_key.lower() and isinstance(sub_value, (int, float)):
                                stats[f"{key}_{sub_key}"] = sub_value
                
                if stats:
                    summary_stats[script_name] = stats
    
    # Combine all data
    output_data = {
        "metadata": metadata,
        "summary_statistics": summary_stats,
        "detailed_results": results
    }
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Results saved to: {output_file}")
        print(f"📊 Summary:")
        print(f"   - Total evaluations: {metadata['total_evaluations']}")
        print(f"   - Successful: {metadata['successful_evaluations']}")
        print(f"   - Failed: {metadata['failed_evaluations']}")
        print(f"   - Model: {metadata['model_name']}")
        print(f"   - Datasets: {metadata['datasets_evaluated']}")
        print(f"   - Model directory: {model_output_dir}")
        
        # Print key metrics
        if summary_stats:
            print(f"\n📈 Key Metrics:")
            for eval_name, stats in summary_stats.items():
                print(f"   {eval_name}:")
                for metric, value in stats.items():
                    if isinstance(value, float) and 'accuracy' in metric.lower():
                        print(f"     - {metric}: {value:.4f}")
                    else:
                        print(f"     - {metric}: {value}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Failed to save results to file: {e}")
        return None

def run_all(args):
    """Run all evaluation scripts"""
    # Define mapping between dataset names and script files
    dataset_scripts = {
        "docvqa": "eval_docvqa_textvqa_chartqa.py",
        "textvqa": "eval_docvqa_textvqa_chartqa.py",
        "chartqa": "eval_docvqa_textvqa_chartqa.py",
        "mme": "eval_mme.py", 
        "ocrbenchv2": "eval_ocrbenchv2.py",
        "mmlu": "eval_mmlu.py",
    }
    
    eval_tasks = []  # List of (script, datasets_for_script) tuples
    
    # Parse datasets argument
    if args.datasets.lower() == "all":
        # Add all available scripts with their respective datasets
        eval_tasks = [
            ("eval_docvqa_textvqa_chartqa.py", "textvqa,docvqa,chartqa"),
            ("eval_mme.py", None),
            ("eval_ocrbenchv2.py", None),
            ("eval_mmlu.py", None)
        ]
    else:
        # Parse comma-separated dataset names
        requested_datasets = [d.strip().lower() for d in args.datasets.split(",")]
        
        # Group datasets by their corresponding scripts
        script_datasets = {}
        for dataset in requested_datasets:
            if dataset in dataset_scripts:
                script = dataset_scripts[dataset]
                if script not in script_datasets:
                    script_datasets[script] = []
                script_datasets[script].append(dataset)
            else:
                print(f"Warning: Unknown dataset '{dataset}'. Available options: {', '.join(dataset_scripts.keys())}")
        
        # Convert to eval_tasks format
        for script, datasets_list in script_datasets.items():
            if script == "eval_docvqa_textvqa_chartqa.py":
                # For docvqa/textvqa/chartqa script, pass specific datasets
                datasets_str = ",".join(datasets_list)
                eval_tasks.append((script, datasets_str))
            else:
                # For other scripts, no datasets parameter needed
                eval_tasks.append((script, None))
    
    if not eval_tasks:
        print("No evaluation scripts to run")
        return
    
    print(f"Running {len(eval_tasks)} evaluation tasks...")
    print(f"Model path: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    print(f"Sequential mode: {args.sequential}")
    print(f"CUDA_VISIBLE_DEVICES: {args.gpus}")
    print("-" * 50)
    
    results = []
    
    # Run evaluations internally and collect results
    if args.sequential:
        for script, datasets in eval_tasks:
            result = run_evaluation_internal(script, args.model_path, datasets)
            results.append(result)
    else:
        # Run evaluations in parallel using multiprocessing
        with multiprocessing.Pool() as pool:
            tasks = [(script, args.model_path, datasets) for script, datasets in eval_tasks]
            results = pool.starmap(run_evaluation_internal, tasks)
    
    # Print summary of results
    print("-" * 50)
    print("Evaluation Results Summary:")
    for result in results:
        print(f"Script: {result['script']}")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success' and 'results' in result:
            # Handle different result formats
            if result['script'] == 'eval_mmlu.py' and 'mmlu' in result['results']:
                # Special formatting for MMLU results
                mmlu_results = result['results']['mmlu']
                print(f"MMLU Results:")
                print(f"  Overall Accuracy: {mmlu_results.get('overall_accuracy', 'N/A'):.4f}")
                print(f"  Correct/Total: {mmlu_results.get('overall_correct', 'N/A')}/{mmlu_results.get('overall_total', 'N/A')}")
                print(f"  Subjects Evaluated: {mmlu_results.get('subject_count', 'N/A')}")
            else:
                # Default formatting for other results
                if isinstance(result['results'], dict):
                    # Pretty print key metrics
                    for key, value in result['results'].items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    if 'accuracy' in sub_key.lower():
                                        print(f"    {sub_key}: {sub_value:.4f}")
                                    else:
                                        print(f"    {sub_key}: {sub_value}")
                                else:
                                    print(f"    {sub_key}: {sub_value}")
                        elif isinstance(value, (int, float)):
                            if 'accuracy' in key.lower():
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"Results: {json.dumps(result['results'], indent=2)}")
        elif result['status'] == 'error':
            print(f"Error: {result['error']}")
        print("-" * 30)
    
    # Save results to file
    save_results_to_file(results, args)
    
    return results


def fix_autoload_essentials():
    """
    Copy essential configuration files from source model to all merged model image directories.
    This ensures that merged models have all necessary files for auto-loading.
    """
    import shutil
    
    source_path = PROJECT_ROOT / "checkpoints/Images/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle"
    copy_files = ["config.json", "generation_config.json", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"]
    
    dest_path = PROJECT_ROOT / "checkpoints/Images/merged_model"
    
    dest_rename_path = dest_path / "renamed"
    
    if not source_path.exists():
        print(f"❌ Source path does not exist: {source_path}")
        return False
    
    if not dest_path.exists():
        print(f"❌ Destination path does not exist: {dest_path}")
        return False
    
    # Check if all required files exist in source
    missing_files = []
    for file_name in copy_files:
        source_file = source_path / file_name
        if not source_file.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing files in source directory: {missing_files}")
        return False
    
    copied_count = 0
    error_count = 0
    
    # Iterate through each folder in dest_path
    for folder in dest_path.iterdir():
        if folder.is_dir():
            image_dir = folder / "image"
            if image_dir.exists() and image_dir.is_dir():
                print(f"📁 Processing {folder.name}/image/")
                
                # ln -s f"{image_dir}" "{dest_rename_path}/{folder.name}"
                if not dest_rename_path.exists():
                    dest_rename_path.mkdir(parents=True, exist_ok=True)
                symlink_path = dest_rename_path / folder.name
                if not symlink_path.exists():
                    try:
                        symlink_path.symlink_to(image_dir)
                        print(f"  ✓ Created symlink: {symlink_path}")
                    except Exception as e:
                        print(f"  ❌ Failed to create symlink: {e}")
                        error_count += 1
                else:
                    print(f"  ⚠️ Symlink already exists: {symlink_path}")
                # Copy each required file
                for file_name in copy_files:
                    source_file = source_path / file_name
                    dest_file = image_dir / file_name
                    
                    try:
                        shutil.copy2(source_file, dest_file)
                        print(f"  ✓ Copied {file_name}")
                        copied_count += 1
                    except Exception as e:
                        print(f"  ❌ Failed to copy {file_name}: {e}")
                        error_count += 1
            else:
                print(f"⚠️  Skipping {folder.name} (no image subdirectory found)")
    
    print(f"\n📊 Summary:")
    print(f"  - Files copied successfully: {copied_count}")
    print(f"  - Copy errors: {error_count}")
    print(f"  - Directories processed: {len([f for f in dest_path.iterdir() if f.is_dir() and (f / 'image').exists()])}")
    
    return error_count == 0

if __name__ == "__main__":
    # fix_autoload_essentials()
    # sys.exit(0)  # Exit early if fix_autoload_essentials is successful
    results = run_all(args)
    if results:
        print(f"\n🎉 Completed {len(results)} evaluations")