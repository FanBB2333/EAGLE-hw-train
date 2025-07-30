"""
Comprehensive Image Evaluation Script

This script runs multiple image evaluation tasks and saves results to organized files.

Usage Examples:
    # Run MMLU evaluation only
    python eval_image_all.py --datasets mmlu
    
    # Run multiple evaluations
    python eval_image_all.py --datasets mmlu,mme,docvqa
    
    # Run all evaluations with custom model
    python eval_image_all.py --datasets all --model_path /path/to/model
    
    # Use custom output directory
    python eval_image_all.py --datasets mmlu --output_dir ./my_results
    
    # View recent results
    python eval_image_all.py --show-results
    
    # Test MMLU integration
    python eval_image_all.py --test-mmlu

Features:
    - Automatic result saving with timestamps and metadata
    - Support for sequential and parallel execution
    - Detailed result formatting and summary statistics
    - Integration with multiple evaluation scripts
    - Recent results viewing functionality
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
        default="docvqa",
        help="Datasets to evaluate. Options: 'all', 'docvqa', 'mme', 'ocrbenchv2', 'mmlu', or comma-separated list (e.g., 'docvqa,mme')"
    )
    parser.add_argument(
        "--gpus", 
        default="6",
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

def run_evaluation_internal(script_name, model_path):
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
        results = evaluate_with_results(model_path)
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
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "eval_image" / "res_folder" / "all_evaluations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_path)
    # Truncate model name if too long
    if len(model_name) > 30:
        model_name = model_name[:27] + "..."
    
    datasets_str = args.datasets.replace(",", "_")
    if len(datasets_str) > 20:
        datasets_str = datasets_str[:17] + "..."
    
    # Create filename
    filename = f"eval_{model_name}_{datasets_str}_{timestamp}.json"
    output_file = output_dir / filename
    
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
        "mme": "eval_mme.py", 
        "ocrbenchv2": "eval_ocrbenchv2.py",
        "mmlu": "eval_mmlu.py",
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

def test_mmlu_integration():
    """Test MMLU evaluation integration"""
    try:
        from eval_mmlu import evaluate_with_results
        print("✓ MMLU module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import MMLU module: {e}")
        return False
    except Exception as e:
        print(f"✗ MMLU integration test failed: {e}")
        return False

def show_recent_results(limit=5):
    """Show the most recent evaluation results"""
    results_dir = PROJECT_ROOT / "eval_image" / "res_folder" / "all_evaluations"
    
    if not results_dir.exists():
        print("No results directory found.")
        return
    
    # Get all JSON files sorted by modification time
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print("No evaluation results found.")
        return
    
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"📊 Recent Evaluation Results (showing last {min(limit, len(json_files))}):")
    print("-" * 80)
    
    for i, file_path in enumerate(json_files[:limit]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            summary = data.get('summary_statistics', {})
            
            print(f"{i+1}. {file_path.name}")
            print(f"   📅 Date: {metadata.get('timestamp', 'Unknown')}")
            print(f"   🤖 Model: {metadata.get('model_name', 'Unknown')}")
            print(f"   📊 Datasets: {metadata.get('datasets_evaluated', 'Unknown')}")
            print(f"   ✅ Success: {metadata.get('successful_evaluations', 0)}/{metadata.get('total_evaluations', 0)}")
            
            if summary:
                print(f"   📈 Key Results:")
                for eval_name, stats in summary.items():
                    if isinstance(stats, dict):
                        for metric, value in stats.items():
                            if 'accuracy' in metric.lower() and isinstance(value, (int, float)):
                                print(f"      - {eval_name} {metric}: {value:.4f}")
            print("-" * 40)
            
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")

if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if '--test-mmlu' in sys.argv:
            test_mmlu_integration()
            sys.exit(0)
        elif '--show-results' in sys.argv:
            show_recent_results()
            sys.exit(0)
        elif '--help-extended' in sys.argv:
            print("Extended Help:")
            print("  --test-mmlu      Test MMLU integration")
            print("  --show-results   Show recent evaluation results")
            print("  --help-extended  Show this extended help")
            sys.exit(0)
    
    results = run_all(args)
    if results:
        print(f"\n🎉 Completed {len(results)} evaluations")
        print("💡 Tip: Use --show-results to view recent evaluation results")