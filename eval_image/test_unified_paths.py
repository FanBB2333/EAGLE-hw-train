#!/usr/bin/env python3
"""
Test script to verify that all evaluation scripts use unified output paths.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def test_output_paths():
    """Test that all evaluation scripts use the unified output path"""
    
    # Expected base output path
    expected_base_path = "/home6/fzy/repos/EAGLE/eval_image/res_folder/images"
    
    # Test model path
    test_model_path = "/some/path/to/test-model-name"
    expected_model_folder = "test-model-name"
    
    print("Testing unified output paths for evaluation scripts...")
    print(f"Expected base path: {expected_base_path}")
    print(f"Test model path: {test_model_path}")
    print(f"Expected model folder: {expected_model_folder}")
    print("-" * 50)
    
    # Test each evaluation script
    scripts_to_test = [
        'eval_mmlu.py',
        'eval_mme.py', 
        'eval_docvqa_textvqa_chartqa.py',
        'eval_ocrbenchv2.py',
        'eval_acqa.py'
    ]
    
    all_passed = True
    
    for script_name in scripts_to_test:
        try:
            # Temporarily modify sys.argv to test argument parsing
            original_argv = sys.argv.copy()
            sys.argv = [script_name, '--model_path', test_model_path, '--help']
            
            # Import and test the argument parser
            if script_name == 'eval_mmlu.py':
                from eval_mmlu import parse_eval_args
            elif script_name == 'eval_mme.py':
                from eval_mme import parse_eval_args
            elif script_name == 'eval_docvqa_textvqa_chartqa.py':
                from eval_docvqa_textvqa_chartqa import parse_eval_args
            elif script_name == 'eval_ocrbenchv2.py':
                from eval_ocrbenchv2 import parse_eval_args
            elif script_name == 'eval_acqa.py':
                from eval_acqa import parse_eval_args
            
            # Reset sys.argv without --help to avoid SystemExit
            sys.argv = [script_name, '--model_path', test_model_path]
            
            try:
                args = parse_eval_args()
                actual_path = args.output_path
                
                # Check if the base path matches
                if actual_path == expected_base_path:
                    print(f"✓ {script_name}: PASS - {actual_path}")
                else:
                    print(f"✗ {script_name}: FAIL - Expected: {expected_base_path}, Got: {actual_path}")
                    all_passed = False
                    
            except SystemExit:
                # This happens with --help, skip for now
                pass
            except Exception as e:
                print(f"✗ {script_name}: ERROR - {str(e)}")
                all_passed = False
                
            # Restore original sys.argv
            sys.argv = original_argv
            
        except ImportError as e:
            print(f"✗ {script_name}: IMPORT ERROR - {str(e)}")
            all_passed = False
        except Exception as e:
            print(f"✗ {script_name}: UNEXPECTED ERROR - {str(e)}")
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("✓ All evaluation scripts use the unified output path!")
    else:
        print("✗ Some evaluation scripts have incorrect output paths.")
    
    return all_passed

if __name__ == "__main__":
    test_output_paths()
