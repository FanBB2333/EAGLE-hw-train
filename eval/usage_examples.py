#!/usr/bin/env python3
"""
Example script showing how to use the updated eval_video_qwen_mvbench.py
with MVBench_dataset offloading capabilities.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Example usage of the updated evaluation script"""
    
    # Example 1: Standard evaluation
    print("=== Example 1: Standard evaluation ===")
    print("python eval_video_qwen_mvbench.py \\")
    print("    --model_path ./checkpoints/your-model \\")
    print("    --subset action_sequence \\")
    print("    --output_path ./output/results.json")
    print()
    
    # Example 2: Evaluation with dataset offloading
    print("=== Example 2: Evaluation with dataset offloading ===")
    print("python eval_video_qwen_mvbench.py \\")
    print("    --model_path ./checkpoints/your-model \\")
    print("    --subset action_sequence \\")
    print("    --offload_dataset \\")
    print("    --processed_dir ./dataset/MVBench/processed \\")
    print("    --num_segments 8 \\")
    print("    --resolution 224 \\")
    print("    --output_path ./output/results_offloaded.json")
    print()
    
    # Example 3: Using pre-processed dataset
    print("=== Example 3: Using pre-processed dataset ===")
    print("python eval_video_qwen_mvbench.py \\")
    print("    --model_path ./checkpoints/your-model \\")
    print("    --use_processed \\")
    print("    --processed_dir ./dataset/MVBench/processed \\")
    print("    --output_path ./output/results_from_processed.json")
    print()
    
    # Example 4: Distributed evaluation with offloading
    print("=== Example 4: Distributed evaluation with offloading ===")
    print("torchrun --nproc_per_node=2 eval_video_qwen_mvbench.py \\")
    print("    --model_path ./checkpoints/your-model \\")
    print("    --distributed \\")
    print("    --offload_dataset \\")
    print("    --processed_dir ./dataset/MVBench/processed \\")
    print("    --output_path ./output/results_distributed.json")
    print()
    
    # Available subsets
    print("=== Available subsets ===")
    subsets = [
        'action_sequence', 'action_prediction', 'action_antonym',
        'fine_grained_action', 'unexpected_action', 'object_existence',
        'object_interaction', 'object_shuffle', 'moving_direction',
        'action_localization', 'scene_transition', 'action_count',
        'moving_count', 'moving_attribute', 'state_change',
        'fine_grained_pose', 'character_order', 'egocentric_navigation',
        'episodic_reasoning', 'counterfactual_inference'
    ]
    
    for i, subset in enumerate(subsets):
        print(f"{i+1:2d}. {subset}")
    print()
    
    # Key improvements
    print("=== Key improvements ===")
    print("1. Unified dataset loading using MVBench_dataset class")
    print("2. Support for dataset offloading/preprocessing")
    print("3. Automatic video segment extraction and frame saving")
    print("4. Efficient reuse of processed data")
    print("5. Support for bounded video segments")
    print("6. Distributed processing support")
    print("7. Better error handling and logging")
    print()
    
    # File structure after processing
    print("=== Processed dataset structure ===")
    print("processed/")
    print("├── processed_index.json")
    print("├── action_sequence/")
    print("│   ├── video_abc12345/")
    print("│   │   ├── 00001.jpg")
    print("│   │   ├── 00002.jpg")
    print("│   │   └── ...")
    print("│   └── ...")
    print("├── action_prediction/")
    print("└── ...")

if __name__ == "__main__":
    main()
