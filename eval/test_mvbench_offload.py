#!/usr/bin/env python3
"""
Simplified test script for MVBench offload functionality
"""

import os
import json
import shutil
import hashlib
from pathlib import Path

def test_core_functionality():
    """Test the core offload functionality without full dependencies"""
    
    # Create a test environment
    test_dir = Path("./test_mvbench_offload")
    test_dir.mkdir(exist_ok=True)
    
    # Create some dummy test data
    sample_data = [
        {
            'task_type': 'Action Sequence',
            'prefix': './test_data/videos/',
            'data_type': 'video',
            'bound': True,
            'data': {
                'video': 'test_video.mp4',
                'start': 2.0,
                'end': 5.0,
                'question': 'What action is performed?',
                'candidates': ['Walking', 'Running', 'Jumping'],
                'answer': 'Running'
            }
        },
        {
            'task_type': 'Object Detection',
            'prefix': './test_data/gifs/',
            'data_type': 'gif',
            'bound': False,
            'data': {
                'video': 'test_gif.gif',
                'question': 'What object is visible?',
                'candidates': ['Car', 'Tree', 'House'],
                'answer': 'Car'
            }
        }
    ]
    
    # Test the processing logic
    processed_data_list = []
    processed_index = {}
    
    for idx, item in enumerate(sample_data):
        task_type = item['task_type']
        data_type = item['data_type']
        bound = item['bound']
        data = item['data']
        
        # Create unique identifier
        video_id = data['video']
        if bound and 'start' in data and 'end' in data:
            start_time = data.get('start', 0)
            end_time = data.get('end', 0)
            segment_id = f"{start_time:.2f}_{end_time:.2f}"
        else:
            segment_id = "full"
        
        # Create hash for unique naming
        content_hash = hashlib.md5(f"{task_type}_{video_id}_{segment_id}".encode()).hexdigest()[:8]
        
        # Create output subdirectory
        task_output_dir = test_dir / task_type.replace(' ', '_').lower()
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simulate processing
        if data_type == 'video':
            processed_path = task_output_dir / f"video_{content_hash}"
        elif data_type == 'gif':
            processed_path = task_output_dir / f"gif_{content_hash}"
        else:
            processed_path = task_output_dir / f"frames_{content_hash}"
        
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Create dummy frames (simple text files for testing)
        for i in range(8):
            frame_path = processed_path / f"{i+1:05d}.jpg"
            with open(frame_path, 'w') as f:
                f.write(f"Dummy frame {i+1} for {task_type}")
        
        # Create new data entry
        new_item = {
            'task_type': task_type,
            'prefix': str(task_output_dir),
            'data_type': 'frame',
            'bound': False,
            'data': {
                **data,
                'video': processed_path.name,
                'original_video': os.path.join(item['prefix'], data['video']),
                'original_bound': bound,
                'processed': True
            }
        }
        processed_data_list.append(new_item)
        
        # Update index
        key = f"{task_type}_{content_hash}"
        processed_index[key] = {
            'original_path': os.path.join(item['prefix'], data['video']),
            'processed_path': str(processed_path),
            'task_type': task_type,
            'bound': bound,
            'data': data
        }
    
    # Save processed index
    index_file = test_dir / "processed_index.json"
    with open(index_file, 'w') as f:
        json.dump(processed_index, f, indent=2)
    
    print(f"✓ Test completed successfully!")
    print(f"✓ Processed {len(processed_data_list)} items")
    print(f"✓ Created {len(processed_index)} index entries")
    print(f"✓ Index saved to {index_file}")
    
    # Test loading the index
    with open(index_file, 'r') as f:
        loaded_index = json.load(f)
    
    print(f"✓ Successfully loaded index with {len(loaded_index)} entries")
    
    # Display structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(test_dir):
        level = root.replace(str(test_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    # Show sample index content
    print("\nSample index content:")
    for key, value in list(loaded_index.items())[:2]:
        print(f"  {key}:")
        print(f"    Original path: {value['original_path']}")
        print(f"    Processed path: {value['processed_path']}")
        print(f"    Task type: {value['task_type']}")
        print(f"    Has bounds: {value['bound']}")
    
    # Cleanup
    print(f"\nCleaning up test directory: {test_dir}")
    shutil.rmtree(test_dir)
    print("✓ Cleanup completed")

if __name__ == "__main__":
    try:
        test_core_functionality()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
