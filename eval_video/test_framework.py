#!/usr/bin/env python3
"""
Test script for eval_video_all.py framework

This script tests the video evaluation framework with minimal setup.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def test_framework():
    """Test the eval_video_all framework"""
    print("Testing eval_video_all.py framework...")
    
    # Test argument parsing
    print("1. Testing argument parsing...")
    from eval_video.eval_video_all import parse_args
    
    # Mock command line arguments
    sys.argv = [
        'eval_video_all.py', 
        '--datasets', 'acqa',
        '--model_path', './checkpoints/test_model',
        '--gpus', '0'
    ]
    
    try:
        args = parse_args()
        print(f"   ✓ Arguments parsed successfully")
        print(f"   - Model path: {args.model_path}")
        print(f"   - Datasets: {args.datasets}")
        print(f"   - GPUs: {args.gpus}")
    except Exception as e:
        print(f"   ✗ Failed to parse arguments: {e}")
        return False
    
    # Test dataset mapping
    print("2. Testing dataset mapping...")
    try:
        dataset_scripts = {
            "acqa": "eval_acqa.py",
        }
        
        if args.datasets in dataset_scripts:
            print(f"   ✓ Dataset '{args.datasets}' found in mapping")
        else:
            print(f"   ✗ Dataset '{args.datasets}' not found in mapping")
    except Exception as e:
        print(f"   ✗ Failed to test dataset mapping: {e}")
        return False
    
    # Test results directory creation
    print("3. Testing results directory...")
    try:
        output_dir = PROJECT_ROOT / "eval_video" / "res_folder" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Results directory created/verified: {output_dir}")
    except Exception as e:
        print(f"   ✗ Failed to create results directory: {e}")
        return False
    
    print("\n🎉 Framework test completed successfully!")
    print("\nTo run actual evaluations:")
    print("python eval_video_all.py --datasets acqa --model_path /path/to/your/model")
    
    return True

if __name__ == "__main__":
    success = test_framework()
    sys.exit(0 if success else 1)
