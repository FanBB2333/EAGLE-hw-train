"""
Comprehensive Video Evaluation Script

This script runs multiple video evaluation tasks and saves results to organized files.
It performs both inference and evaluation scoring in a two-stage process:

1. **Inference Stage**: Runs model inference to generate predictions
2. **Evaluation Stage**: Calls parse_output functions to calculate metrics and scores

Usage Examples:
    # Run ActivityNetQA evaluation only
    python eval_video_all.py --datasets acqa
    
    # Run eval_video_qwen datasets
    python eval_video_all.py --datasets charades,mvbench,activitynet
    
    # Run all evaluations
    python eval_video_all.py --datasets all
    
    # Use custom model and output directory
    python eval_video_all.py --datasets charades --model_path /path/to/model --output_dir ./my_results
    
    # Only evaluate existing prediction files (no inference)
    python eval_video_all.py --datasets acqa --eval_only
    
    # Evaluate specific datasets from existing files
    python eval_video_all.py --datasets mvbench,charades --eval_only --model_path /path/to/model

Supported Datasets:
    - acqa: ActivityNetQA (from eval_acqa.py)
    - activitynet: ActivityNet Captions (from eval_video_qwen.py)
    - charades: Charades Actions (from eval_video_qwen.py)
    - qvhighlights: QV Highlights (from eval_video_qwen.py)
    - youcook2: YouCook2 (from eval_video_qwen.py)
    - mvbench: MVBench (from eval_video_qwen.py)

Evaluation Process:
    1. For each dataset, the script first runs inference using evaluate_with_results()
    2. Then it calls parse_output() to calculate evaluation metrics (accuracy, mIoU, Recall@K, etc.)
    3. Results include both raw inference outputs and calculated metrics
    4. All results are saved with detailed metadata and summary statistics

Features:
    - Two-stage evaluation: inference + scoring
    - Evaluation-only mode: parse existing prediction files without inference (--eval_only)
    - Automatic result saving with timestamps and metadata in model-specific directories
    - Dataset-specific output directories: eval_video/res_folder/videos/{model_name}/{dataset}/
    - Support for sequential and parallel execution
    - Detailed result formatting and summary statistics
    - Integration with multiple evaluation scripts
    - Comprehensive metrics calculation (accuracy, mIoU, Recall@K)
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
        # default="./checkpoints/Videos/merged_model/finetune-video-llama3.2-3b-merged1-qwen-0.98-0.02",
        default="./checkpoints/Videos/merged_model/finetune-video-llama3.2-3b-merged1-qwen-0.98-0.02-pass5",
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
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Only evaluate existing JSON results without running inference. "
             "Searches for prediction files in the output directory and calculates metrics."
    )
    return parser.parse_args()

# Parse arguments and set GPU environment early
args = parse_args()

# Preprocess model_path: convert to absolute path relative to PROJECT_ROOT if not already absolute
if not os.path.isabs(args.model_path):
    args.model_path = str(PROJECT_ROOT / args.model_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

def run_evaluation_only(script_name, model_path, datasets=None, output_path=None):
    """Run evaluation only on existing prediction files"""
    try:
        # Import parse_output functions dynamically when needed
        if script_name == "eval_acqa.py":
            from eval_acqa import parse_output
        elif script_name == "eval_video_qwen.py":
            from eval_video_qwen import parse_output
        else:
            return {
                'script': script_name,
                'status': 'error',
                'error': f'Unknown script: {script_name}'
            }
        
        print(f"Starting evaluation-only for {script_name}...")
        
        # Calculate output path
        model_name = os.path.basename(model_path.rstrip('/'))
        if output_path:
            base_output_dir = Path(output_path) / model_name
        else:
            base_output_dir = PROJECT_ROOT / "eval_video" / "res_folder" / "videos" / model_name
        
        # Create args object for parse_output
        import argparse
        args = argparse.Namespace()
        args.model_path = model_path
        args.output_path = str(base_output_dir)
        
        # Check if output directory exists
        if not base_output_dir.exists():
            return {
                'script': script_name,
                'status': 'error',
                'error': f'Output directory does not exist: {base_output_dir}'
            }
        
        # Look for prediction files
        prediction_files = []
        if script_name == "eval_acqa.py":
            # Try multiple possible file locations for acqa
            possible_acqa_files = [
                base_output_dir / "acqa.json",
                base_output_dir / "acqa" / "acqa.json",
                base_output_dir / model_name / "acqa.json"
            ]
            for acqa_file in possible_acqa_files:
                if acqa_file.exists():
                    prediction_files.append(str(acqa_file))
                    break
        elif script_name == "eval_video_qwen.py":
            # Look for video prediction files in multiple possible formats
            for dataset in datasets or ['activitynet', 'charades', 'qvhighlights', 'youcook2', 'mvbench']:
                # Try multiple possible file locations and naming conventions
                possible_files = [
                    base_output_dir / f"{dataset}.json",
                    base_output_dir / f"{dataset}_output.json",
                    base_output_dir / dataset / f"{dataset}.json",
                    base_output_dir / model_name / f"{dataset}.json"
                ]
                for pred_file in possible_files:
                    if pred_file.exists():
                        prediction_files.append(str(pred_file))
                        break
        
        if not prediction_files:
            # Create detailed error message showing what files were searched for
            searched_locations = []
            if script_name == "eval_acqa.py":
                searched_locations = [
                    str(base_output_dir / "acqa.json"),
                    str(base_output_dir / "acqa" / "acqa.json"),
                    str(base_output_dir / model_name / "acqa.json")
                ]
            elif script_name == "eval_video_qwen.py":
                for dataset in datasets or ['activitynet', 'charades', 'qvhighlights', 'youcook2', 'mvbench']:
                    searched_locations.extend([
                        str(base_output_dir / f"{dataset}.json"),
                        str(base_output_dir / f"{dataset}_output.json"),
                        str(base_output_dir / dataset / f"{dataset}.json"),
                        str(base_output_dir / model_name / f"{dataset}.json")
                    ])
            
            return {
                'script': script_name,
                'status': 'error',
                'error': f'No prediction files found in {base_output_dir}. Searched locations: {searched_locations[:10]}...' if len(searched_locations) > 10 else f'No prediction files found. Searched locations: {searched_locations}'
            }
        
        print(f"Found prediction files: {prediction_files}")
        
        try:
            # Call parse_output to get evaluation metrics
            # For eval_only mode, we need to construct a proper evaluation_results structure
            if script_name == "eval_acqa.py":
                # For eval_acqa, pass None to force loading from files
                evaluation_metrics = parse_output(evaluation_results=None, args=args)
            elif script_name == "eval_video_qwen.py":
                # For eval_video_qwen, construct a proper evaluation_results structure
                # Since parse_output expects specific structure, we need to create it
                mock_evaluation_results = {
                    'status': 'success',
                    'total_tasks': len(prediction_files),
                    'successful_tasks': [],
                    'failed_tasks': [],
                    'success_rate': 0.0,
                    'detailed_results': {}
                }
                
                # Process each prediction file to build the structure
                for pred_file in prediction_files:
                    file_path = Path(pred_file)
                    task_name = file_path.stem.replace('_output', '')  # Remove _output suffix
                    
                    try:
                        with open(pred_file, 'r') as f:
                            predictions = json.load(f)
                        
                        mock_evaluation_results['successful_tasks'].append(task_name)
                        mock_evaluation_results['detailed_results'][task_name] = {
                            'status': 'success',
                            'total_predictions': len(predictions) if isinstance(predictions, list) else 1,
                            'output_file': pred_file,
                            'task_specific_info': {
                                'description': f'Loaded from existing prediction file: {pred_file}'
                            }
                        }
                    except Exception as e:
                        mock_evaluation_results['failed_tasks'].append(task_name)
                        mock_evaluation_results['detailed_results'][task_name] = {
                            'status': 'error',
                            'error': f'Failed to load prediction file {pred_file}: {str(e)}'
                        }
                
                # Update success rate
                total_tasks = len(mock_evaluation_results['successful_tasks']) + len(mock_evaluation_results['failed_tasks'])
                if total_tasks > 0:
                    mock_evaluation_results['success_rate'] = len(mock_evaluation_results['successful_tasks']) / total_tasks
                    mock_evaluation_results['total_tasks'] = total_tasks
                
                evaluation_metrics = parse_output(evaluation_results=mock_evaluation_results, args=args)
            else:
                evaluation_metrics = parse_output(evaluation_results=None, args=args)
                
            print(f"✓ {script_name} evaluation completed successfully")
            
            # Create a mock inference results structure for compatibility
            mock_inference_results = {
                'status': 'loaded_from_files',
                'prediction_files': prediction_files,
                'output_path': str(base_output_dir),
                'eval_only_mode': True
            }
            
            # Combine with evaluation metrics
            combined_results = {
                'inference_results': mock_inference_results,
                'evaluation_metrics': evaluation_metrics,
                'status': 'completed',
                'eval_only_mode': True
            }
            
            return {
                'script': script_name,
                'status': 'success',
                'results': combined_results
            }
            
        except Exception as e:
            print(f"✗ {script_name} evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'script': script_name,
                'status': 'error',
                'error': f'Evaluation failed: {str(e)}'
            }
        
    except Exception as e:
        print(f"✗ Failed to run evaluation-only for {script_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'script': script_name,
            'status': 'error',
            'error': str(e)
        }

def run_evaluation_internal(script_name, model_path, datasets=None):
    """Run evaluation internally and return results"""
    try:
        # Import evaluation functions dynamically when needed
        if script_name == "eval_acqa.py":
            from eval_acqa import evaluate_with_results, parse_output
        elif script_name == "eval_video_qwen.py":
            from eval_video_qwen import evaluate_with_results, parse_output
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
        inference_results = evaluate_with_results(model_path, datasets, str(base_output_dir))
        
        print(f"✓ {script_name} inference completed successfully")
        
        # Now call parse_output to evaluate the inference results
        print(f"Starting evaluation/scoring for {script_name}...")
        
        # Create args object for parse_output
        import argparse
        args = argparse.Namespace()
        args.model_path = model_path
        args.output_path = str(base_output_dir)
        
        try:
            # Call parse_output to get evaluation metrics
            evaluation_metrics = parse_output(evaluation_results=inference_results, args=args)
            print(f"✓ {script_name} evaluation/scoring completed successfully")
            
            # Combine inference results with evaluation metrics
            combined_results = {
                'inference_results': inference_results,
                'evaluation_metrics': evaluation_metrics,
                'status': 'completed'
            }
            
            return {
                'script': script_name,
                'status': 'success',
                'results': combined_results
            }
            
        except Exception as e:
            print(f"⚠ {script_name} inference completed but evaluation failed: {str(e)}")
            # Return inference results even if evaluation fails
            return {
                'script': script_name,
                'status': 'partial_success',
                'results': inference_results,
                'evaluation_error': str(e)
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
        "eval_only_mode": args.eval_only,
        "gpu_devices": args.gpus,
        "total_evaluations": len(results),
        "successful_evaluations": len([r for r in results if r['status'] == 'success']),
        "partial_success_evaluations": len([r for r in results if r['status'] == 'partial_success']),
        "failed_evaluations": len([r for r in results if r['status'] == 'error']),
        "evaluation_process": "evaluation_only" if args.eval_only else "two_stage_inference_and_scoring",
        "evaluation_description": "Parse existing prediction files for metrics" if args.eval_only else "Runs inference then calls parse_output for metric calculation"
    }
    
    # Prepare summary statistics
    summary_stats = {}
    for result in results:
        if result['status'] in ['success', 'partial_success'] and 'results' in result:
            script_name = result['script'].replace('.py', '')
            
            # Handle new combined format with evaluation metrics
            if isinstance(result['results'], dict) and 'evaluation_metrics' in result['results']:
                evaluation_metrics = result['results']['evaluation_metrics']
                inference_results = result['results']['inference_results']
                
                if script_name == 'eval_acqa':
                    # Extract ActivityNetQA evaluation metrics
                    summary_stats[script_name] = {
                        "accuracy": evaluation_metrics.get('accuracy'),
                        "total_questions": evaluation_metrics.get('total_questions'),
                        "correct_answers": evaluation_metrics.get('correct_answers'),
                        "valid_predictions": evaluation_metrics.get('valid_predictions'),
                        "output_file": inference_results.get('output_file')
                    }
                elif script_name == 'eval_video_qwen':
                    # Extract video evaluation metrics
                    summary_stats[script_name] = {
                        "total_tasks": evaluation_metrics.get('total_tasks', 0),
                        "successful_task_count": evaluation_metrics.get('successful_task_count', 0),
                        "failed_task_count": evaluation_metrics.get('failed_task_count', 0),
                        "success_rate": evaluation_metrics.get('success_rate', 0.0)
                    }
                    
                    # Add individual task metrics
                    if 'task_details' in evaluation_metrics:
                        for task_name, task_info in evaluation_metrics['task_details'].items():
                            if task_info.get('status') == 'success':
                                task_stats = {}
                                if task_name == 'mvbench':
                                    task_stats = {
                                        "accuracy": task_info.get('accuracy', 0.0),
                                        "correct_predictions": task_info.get('correct_predictions', 0),
                                        "total_predictions": task_info.get('total_predictions', 0)
                                    }
                                else:
                                    task_stats = {
                                        "mIoU": task_info.get('mIoU', 0.0),
                                        "Recall": task_info.get('Recall', {}),
                                        "valid_samples": task_info.get('valid_samples', 0),
                                        "invalid_predictions": task_info.get('invalid_predictions', 0),
                                        "total_predictions": task_info.get('total_predictions', 0)
                                    }
                                summary_stats[f"{script_name}_{task_name}"] = task_stats
                            
            # Handle legacy formats for backward compatibility
            elif script_name == 'eval_acqa' and isinstance(result['results'], dict):
                # Extract ActivityNetQA specific stats (legacy format)
                acqa_data = result['results']
                summary_stats[script_name] = {
                    "total_questions": acqa_data.get('total_questions'),
                    "valid_predictions": acqa_data.get('valid_predictions'),
                    "output_file": acqa_data.get('output_file')
                }
            elif script_name == 'eval_video_qwen' and isinstance(result['results'], dict):
                # Extract eval_video_qwen specific stats (legacy format)
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
        print(f"   - Partial success: {metadata['partial_success_evaluations']}")
        print(f"   - Failed: {metadata['failed_evaluations']}")
        print(f"   - Model: {metadata['model_name']}")
        print(f"   - Datasets: {metadata['datasets_evaluated']}")
        print(f"   - Process: {metadata['evaluation_process']}")
        
        # Print key metrics
        if summary_stats:
            print(f"\n📈 Key Metrics:")
            for eval_name, stats in summary_stats.items():
                print(f"   {eval_name}:")
                for metric, value in stats.items():
                    if isinstance(value, float):
                        if 'accuracy' in metric.lower() or 'miou' in metric.lower():
                            print(f"     - {metric}: {value:.4f}")
                        elif 'rate' in metric.lower():
                            print(f"     - {metric}: {value:.2%}")
                        else:
                            print(f"     - {metric}: {value:.4f}")
                    elif isinstance(value, dict) and metric == 'Recall':
                        recall_str = ", ".join([f"R@{k}: {v:.4f}" for k, v in value.items()])
                        print(f"     - {metric}: {recall_str}")
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
    print(f"Eval only mode: {args.eval_only}")
    print(f"CUDA_VISIBLE_DEVICES: {args.gpus}")
    print("-" * 50)
    
    results = []
    
    # Choose evaluation mode based on --eval_only flag
    if args.eval_only:
        print("🔍 Running in evaluation-only mode (parsing existing prediction files)...")
        # Run evaluation-only mode
        if args.sequential:
            for script, datasets in eval_tasks:
                result = run_evaluation_only(script, args.model_path, datasets, args.output_dir)
                results.append(result)
        else:
            # Run evaluations in parallel using multiprocessing
            with multiprocessing.Pool() as pool:
                tasks = [(script, args.model_path, datasets, args.output_dir) for script, datasets in eval_tasks]
                results = pool.starmap(run_evaluation_only, tasks)
    else:
        print("🚀 Running full evaluation mode (inference + scoring)...")
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
            # Check if this is eval_only mode
            is_eval_only = result['results'].get('eval_only_mode', False)
            if is_eval_only:
                print(f"🔍 Evaluation-only mode (parsed existing files)")
            
            # Handle new combined results format with inference + evaluation
            if isinstance(result['results'], dict) and 'evaluation_metrics' in result['results']:
                print(f"📊 Evaluation Metrics:")
                evaluation_metrics = result['results']['evaluation_metrics']
                
                if result['script'] == 'eval_acqa.py':
                    # ActivityNetQA specific metrics
                    if 'accuracy' in evaluation_metrics:
                        print(f"  ActivityNetQA Accuracy: {evaluation_metrics['accuracy']:.4f}%")
                    if 'total_questions' in evaluation_metrics:
                        print(f"  Total Questions: {evaluation_metrics['total_questions']}")
                    if 'correct_answers' in evaluation_metrics:
                        print(f"  Correct Answers: {evaluation_metrics['correct_answers']}")
                    if 'valid_predictions' in evaluation_metrics:
                        print(f"  Valid Predictions: {evaluation_metrics['valid_predictions']}")
                        
                elif result['script'] == 'eval_video_qwen.py':
                    # Video evaluation specific metrics
                    if 'task_details' in evaluation_metrics:
                        print(f"  Video Task Results:")
                        for task_name, task_info in evaluation_metrics['task_details'].items():
                            status_icon = "✓" if task_info.get('status') == 'success' else "✗"
                            predictions = task_info.get('total_predictions', 'N/A')
                            
                            if task_info.get('status') == 'success':
                                if task_name == 'mvbench':
                                    accuracy = task_info.get('accuracy', 0.0)
                                    correct = task_info.get('correct_predictions', 0)
                                    print(f"    {status_icon} {task_name}: Accuracy {accuracy:.4f} ({correct}/{predictions})")
                                else:
                                    miou = task_info.get('mIoU', 0.0)
                                    valid_samples = task_info.get('valid_samples', 0)
                                    invalid_samples = task_info.get('invalid_predictions', 0)
                                    recall = task_info.get('Recall', {})
                                    recall_str = ", ".join([f"R@{k}: {v:.4f}" for k, v in recall.items()]) if recall else "N/A"
                                    print(f"    {status_icon} {task_name}: mIoU {miou:.4f}, {recall_str}")
                                    print(f"        Valid: {valid_samples}, Invalid: {invalid_samples}, Total: {predictions}")
                            else:
                                error = task_info.get('error', 'Unknown error')
                                print(f"    {status_icon} {task_name}: Failed - {error}")
                    
                    if 'successful_task_count' in evaluation_metrics:
                        total_tasks = evaluation_metrics.get('total_tasks', 0)
                        successful = evaluation_metrics['successful_task_count']
                        print(f"  Overall Success Rate: {successful}/{total_tasks} tasks")
                        
            # Handle legacy result formats for backward compatibility
            elif result['script'] == 'eval_acqa.py' and isinstance(result['results'], dict):
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
                    
        elif result['status'] == 'partial_success':
            print(f"⚠ Inference completed but evaluation failed")
            print(f"Evaluation Error: {result.get('evaluation_error', 'Unknown error')}")
            # Still show inference results if available
            if 'results' in result:
                print(f"Inference completed successfully with {result['results'].get('total_questions', 'N/A')} items")
                
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
