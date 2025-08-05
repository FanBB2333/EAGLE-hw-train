"""
Comprehensive Video Evaluation Script

This script runs multiple video evaluation tasks and saves results to organized files.

Usage Examples:
    # Run ActivityNetQA evaluation only
    python eval_video_all.py --datasets acqa
    
    # Run eval_video_qwen datasets
    python eval_video_all.py --datasets charades,mvbench,activitynet
    
    # Run all evaluations
    python eval_video_all.py --datasets all
    
    # Use custom model and output directory
    python eval_video_all.py --datasets charades --model_path /path/to/model --output_dir ./my_results

Supported Datasets:
    - acqa: ActivityNetQA (from eval_acqa.py)
    - activitynet: ActivityNet Captions (from eval_video_qwen.py)
    - charades: Charades Actions (from eval_video_qwen.py)
    - qvhighlights: QV Highlights (from eval_video_qwen.py)
    - youcook2: YouCook2 (from eval_video_qwen.py)
    - mvbench: MVBench (from eval_video_qwen.py)

Features:
    - Automatic result saving with timestamps and metadata in model-specific directories
    - Dataset-specific output directories: eval_video/res_folder/videos/{model_name}/{dataset}/
    - Support for sequential and parallel execution
    - Detailed result formatting and summary statistics
    - Integration with multiple evaluation scripts
    - Extensible framework for adding new video evaluation datasets
"""

import os
import multiprocessing
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add parent directory to path
from pathlib import Path
import json
from train_video1 import ModelArguments
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Parse arguments first to set environment variables early
def parse_args():
    parser = argparse.ArgumentParser(description="Run all video evaluation tasks")
    parser.add_argument(
        "--model_path", 
        default="./checkpoints/Videos/merged_model/finetune-video-llama3.2-3b-merged1-qwen-0.98-0.02",
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
        default="acqa",
        help="Datasets to evaluate. Options: 'all', 'acqa', or comma-separated list. "
             "Available datasets: acqa (ActivityNetQA), activitynet, charades, "
             "qvhighlights, youcook2, mvbench"
    )
    parser.add_argument(
        "--gpus", 
        default="3",
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
        if script_name == "eval_acqa.py":
            from eval_acqa import evaluate_with_results
        elif script_name == "eval_video_qwen.py":
            from eval_video_qwen import evaluate_with_results
        else:
            return {
                'script': script_name,
                'status': 'error',
                'error': f'Unknown script: {script_name}'
            }
        
        # Call the evaluation function directly
        print(f"Starting evaluation for {script_name}...")
        
        # Calculate output path for both scripts to use unified structure
        model_name = os.path.basename(model_path.rstrip('/'))
        base_output_dir = PROJECT_ROOT / "eval_video" / "res_folder" / "videos" / model_name
        
        # Call the evaluation function with output_path
        results = evaluate_with_results(model_path, datasets, str(base_output_dir))
        
        print(f"✓ {script_name} completed successfully")
        
        # Handle different return formats
        if isinstance(results, dict):
            if 'status' in results and results['status'] == 'completed':
                # New format with status field
                return {
                    'script': script_name,
                    'status': 'success',
                    'results': results
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
                    'results': results
                }
        else:
            # Unexpected format
            return {
                'script': script_name,
                'status': 'success',
                'results': {'raw_output': results}
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
    
    # Create dataset-specific output directory for this evaluation
    model_name = os.path.basename(args.model_path.rstrip('/'))
    
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = PROJECT_ROOT / "eval_video" / "res_folder" / "videos" / model_name
    
    # Create dataset-specific subdirectory
    datasets_str = args.datasets.replace(",", "_")
    dataset_output_dir = base_output_dir / datasets_str
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create simplified filename since directory structure now contains model and dataset info
    filename = f"eval_results_{timestamp}.json"
    output_file = dataset_output_dir / filename
    
    # Prepare metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_path": args.model_path,
        "model_name": model_name,
        "datasets_evaluated": args.datasets,
        "output_directory": str(dataset_output_dir),
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
            
            if script_name == 'eval_acqa' and isinstance(result['results'], dict):
                # Extract ActivityNetQA specific stats
                acqa_data = result['results']
                summary_stats[script_name] = {
                    "total_questions": acqa_data.get('total_questions'),
                    "valid_predictions": acqa_data.get('valid_predictions'),
                    "output_file": acqa_data.get('output_file')
                }
            elif script_name == 'eval_video_qwen' and isinstance(result['results'], dict):
                # Extract eval_video_qwen specific stats
                qwen_data = result['results']
                summary_stats[script_name] = {
                    "total_tasks": qwen_data.get('total_tasks', 0),
                    "successful_tasks": len(qwen_data.get('successful_tasks', [])),
                    "failed_tasks": len(qwen_data.get('failed_tasks', [])),
                    "success_rate": qwen_data.get('success_rate', 0.0),
                    "tasks_evaluated": qwen_data.get('successful_tasks', [])
                }
                
                # Add individual task statistics if available
                if 'summary_statistics' in qwen_data:
                    for task, task_stats in qwen_data['summary_statistics'].items():
                        summary_stats[f"{script_name}_{task}"] = task_stats
                        
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
        print(f"   - Output directory: {dataset_output_dir}")
        print(f"   - Total evaluations: {metadata['total_evaluations']}")
        print(f"   - Successful: {metadata['successful_evaluations']}")
        print(f"   - Failed: {metadata['failed_evaluations']}")
        print(f"   - Model: {metadata['model_name']}")
        print(f"   - Datasets: {metadata['datasets_evaluated']}")
        
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
        "acqa": "eval_acqa.py",
        # eval_video_qwen datasets
        "activitynet": "eval_video_qwen.py",
        "charades": "eval_video_qwen.py",
        "qvhighlights": "eval_video_qwen.py",
        "youcook2": "eval_video_qwen.py",
        "mvbench": "eval_video_qwen.py",
    }
    
    eval_tasks = []  # List of (script, datasets_for_script) tuples
    
    # Parse datasets argument
    if args.datasets.lower() == "all":
        # Add all available scripts with their respective datasets
        eval_tasks = [
            ("eval_acqa.py", ["acqa"]),
            ("eval_video_qwen.py", ["activitynet", "charades", "qvhighlights", "youcook2", "mvbench"]),
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
            eval_tasks.append((script, datasets_list))
    
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
            if result['script'] == 'eval_acqa.py' and isinstance(result['results'], dict):
                # Special formatting for ActivityNetQA results
                acqa_results = result['results']
                print(f"ActivityNetQA Results:")
                print(f"  Total Questions: {acqa_results.get('total_questions', 'N/A')}")
                print(f"  Valid Predictions: {acqa_results.get('valid_predictions', 'N/A')}")
                print(f"  Output File: {acqa_results.get('output_file', 'N/A')}")
                
                # Show sample predictions
                if 'sample_predictions' in acqa_results:
                    print(f"  Sample Predictions:")
                    for i, sample in enumerate(acqa_results['sample_predictions'][:3]):
                        print(f"    {i+1}. Q: {sample['question'][:100]}...")
                        print(f"       A: {sample['answer']}")
                        print(f"       P: {sample['prediction']}")
                        
            elif result['script'] == 'eval_video_qwen.py' and isinstance(result['results'], dict):
                # Special formatting for eval_video_qwen results
                qwen_results = result['results']
                print(f"Video Qwen Evaluation Results:")
                print(f"  Total Tasks: {qwen_results.get('total_tasks', 'N/A')}")
                print(f"  Successful Tasks: {len(qwen_results.get('successful_tasks', []))}")
                print(f"  Failed Tasks: {len(qwen_results.get('failed_tasks', []))}")
                print(f"  Success Rate: {qwen_results.get('success_rate', 0):.2%}")
                
                # Show successful tasks
                if qwen_results.get('successful_tasks'):
                    print(f"  Completed Tasks: {', '.join(qwen_results['successful_tasks'])}")
                
                # Show failed tasks if any
                if qwen_results.get('failed_tasks'):
                    print(f"  Failed Tasks: {', '.join(qwen_results['failed_tasks'])}")
                
                # Show detailed results for each task
                if 'detailed_results' in qwen_results:
                    for task, task_data in qwen_results['detailed_results'].items():
                        if isinstance(task_data, dict) and task_data.get('status') == 'success':
                            predictions = task_data.get('total_predictions', 0)
                            print(f"    {task}: {predictions} predictions")
                            
                            # Show sample prediction
                            if 'sample_predictions' in task_data and task_data['sample_predictions']:
                                sample = task_data['sample_predictions'][0]
                                print(f"      Sample Q: {sample.get('question', 'N/A')[:80]}...")
                                print(f"      Sample P: {sample.get('prediction', 'N/A')[:80]}...")
                        elif isinstance(task_data, dict) and task_data.get('status') == 'error':
                            print(f"    {task}: ERROR - {task_data.get('error', 'Unknown error')}")
                            
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

if __name__ == "__main__":
    results = run_all(args)
    if results:
        print(f"\n🎉 Completed {len(results)} evaluations")
