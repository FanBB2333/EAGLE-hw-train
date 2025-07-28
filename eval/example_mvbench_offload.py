#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved MVBench_dataset class
with dataset offloading capabilities.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_video_qwen_mvbench import MVBench_dataset, data_list, data_dir

def main():
    """Example of using the MVBench dataset with offloading"""
    
    # Step 1: Create original dataset
    print("Creating original MVBench dataset...")
    dataset = MVBench_dataset(data_dir, data_list, num_segments=8, resolution=224)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Dataset info:\n{dataset}")
    
    # Step 2: Offload dataset (process and save all bounded videos)
    print("\nOffloading dataset...")
    processed_dir = "./dataset/MVBench/processed"
    processed_index = dataset.offload_dataset(
        output_dir=processed_dir,
        num_frames=8
    )
    print(f"Processed {len(processed_index)} video segments")
    
    # Step 3: Create new dataset from processed data
    print("\nCreating dataset from processed data...")
    processed_dataset = MVBench_dataset.from_processed(
        data_dir, data_list, processed_dir, num_segments=8, resolution=224
    )
    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Step 4: Compare sample data
    print("\nComparing sample data...")
    if len(dataset) > 0 and len(processed_dataset) > 0:
        # Get first sample from both datasets
        original_sample = dataset[0]
        processed_sample = processed_dataset[0]
        
        print("Original sample keys:", original_sample.keys())
        print("Processed sample keys:", processed_sample.keys())
        print("Original video shape:", original_sample['video'].shape)
        print("Processed video shape:", processed_sample['video'].shape)
        print("Same question?", original_sample['question'] == processed_sample['question'])
        print("Same answer?", original_sample['answer'] == processed_sample['answer'])
    
    # Step 5: Show processing statistics
    print("\nProcessing statistics:")
    task_counts = {}
    for item in processed_dataset.data_list:
        task_type = item['task_type']
        if task_type not in task_counts:
            task_counts[task_type] = 0
        task_counts[task_type] += 1
    
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} samples")


def test_loading_processed():
    """Test loading an existing processed dataset"""
    print("\nTesting loading of existing processed dataset...")
    
    processed_dir = "./dataset/MVBench/processed"
    processed_index_file = os.path.join(processed_dir, "processed_index.json")
    
    if os.path.exists(processed_index_file):
        # Load processed dataset
        dataset = MVBench_dataset(data_dir, data_list)
        success = dataset.load_processed_dataset(processed_index_file)
        
        if success:
            print(f"Successfully loaded processed dataset with {len(dataset)} items")
            
            # Test getting a sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"Sample video shape: {sample['video'].shape}")
                print(f"Sample task type: {sample['task_type']}")
        else:
            print("Failed to load processed dataset")
    else:
        print(f"Processed index file not found: {processed_index_file}")


if __name__ == "__main__":
    try:
        main()
        test_loading_processed()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
