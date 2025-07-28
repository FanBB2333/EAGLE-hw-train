# MVBench Dataset Offloading

This document describes the enhanced `MVBench_dataset` class with dataset offloading capabilities.

## Overview

The enhanced `MVBench_dataset` class now includes the ability to preprocess and save video segments to improve loading efficiency. This is particularly useful for bounded video segments that need to be cropped and processed repeatedly.

## Key Features

1. **Dataset Offloading**: Process and save bounded video segments as frames
2. **Multiple Format Support**: Handle video files, GIF files, and frame directories
3. **Efficient Storage**: Save processed frames in organized directory structure
4. **Index Management**: Maintain mapping between original and processed data
5. **Easy Reloading**: Quickly load processed datasets for subsequent runs

## Usage

### 1. Basic Dataset Creation

```python
from eval_video_qwen_mvbench import MVBench_dataset, data_list, data_dir

# Create original dataset
dataset = MVBench_dataset(data_dir, data_list, num_segments=8, resolution=224)
```

### 2. Offload Dataset (Process and Save)

```python
# Process all videos and save frames
processed_index = dataset.offload_dataset(
    output_dir="./dataset/MVBench/processed",
    num_frames=8
)
```

### 3. Load Processed Dataset

```python
# Method 1: Create from processed data
processed_dataset = MVBench_dataset.from_processed(
    data_dir, data_list, 
    processed_dir="./dataset/MVBench/processed",
    num_segments=8, resolution=224
)

# Method 2: Load into existing dataset
dataset = MVBench_dataset(data_dir, data_list)
dataset.load_processed_dataset("./dataset/MVBench/processed/processed_index.json")
```

## Directory Structure

After offloading, the processed directory structure will be:

```
processed/
├── processed_index.json          # Index mapping original to processed
├── action_sequence/
│   ├── video_abc12345/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── video_def67890/
│       └── ...
├── action_prediction/
│   └── ...
└── ...
```

## Methods

### `offload_dataset(output_dir, num_frames)`

Process all videos in the dataset and save extracted frames.

**Parameters:**
- `output_dir` (str): Directory to save processed data
- `num_frames` (int): Number of frames to extract per segment

**Returns:**
- `dict`: Index mapping of processed data

### `load_processed_dataset(processed_index_file)`

Load previously processed dataset from index file.

**Parameters:**
- `processed_index_file` (str): Path to processed index JSON file

**Returns:**
- `bool`: Success status

### `from_processed(data_dir, data_list, processed_dir, ...)`

Class method to create dataset instance from processed data.

**Parameters:**
- `data_dir` (str): Original data directory
- `data_list` (dict): Original data list configuration
- `processed_dir` (str): Directory containing processed data
- Other parameters same as `__init__`

## Processing Details

### Video Processing
- Uses `decord.VideoReader` to read video files
- Extracts frames based on temporal bounds (start/end times)
- Saves frames as numbered JPEG files

### GIF Processing
- Uses `imageio` to read GIF files
- Converts RGBA to RGB format
- Extracts frames based on temporal bounds

### Frame Processing
- Copies specific frames from frame directories
- Maintains original numbering scheme
- Supports bounded frame selection

## Benefits

1. **Faster Loading**: Processed frames load much faster than video decoding
2. **Consistent Format**: All data types converted to frame sequences
3. **Reduced I/O**: Eliminates repeated video processing
4. **Disk Space Optimization**: Only saves necessary frames
5. **Easy Distribution**: Processed datasets can be easily shared

## Error Handling

The implementation includes comprehensive error handling:
- Missing video files are skipped with warnings
- Processing errors are logged and don't stop the entire process
- Corrupted data is detected and excluded
- File system permissions are checked

## Example Script

See `example_mvbench_offload.py` for a complete usage example.

## Requirements

- `decord` for video processing
- `imageio` for GIF processing
- `PIL` for image operations
- `tqdm` for progress tracking
- Standard libraries: `os`, `shutil`, `hashlib`, `json`
