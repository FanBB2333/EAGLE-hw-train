# MVBench Evaluation Script

This script is specifically designed for evaluating models on the MVBench dataset using Hugging Face's datasets library.

## Features

- Uses `load_dataset("OpenGVLab/MVBench", subset_name)` to automatically load MVBench data
- **Evaluates all 20 MVBench subsets by default**
- Supports both single-process and distributed evaluation
- **Calculates accuracy metrics grouped by subset**
- Handles missing video files gracefully
- **Outputs detailed summary with per-subset accuracy**
- Customizable output paths and model configurations

## MVBench Subsets

The script supports all 20 MVBench subsets:
- `action_sequence`
- `moving_count` 
- `action_prediction`
- `episodic_reasoning`
- `action_antonym`
- `action_count`
- `scene_transition`
- `object_shuffle`
- `object_existence`
- `fine_grained_pose`
- `unexpected_action`
- `moving_direction`
- `state_change`
- `object_interaction`
- `character_order`
- `action_localization`
- `counterfactual_inference`
- `fine_grained_action`
- `moving_attribute`
- `egocentric_navigation`

## Usage

### Evaluate All Subsets (Recommended)

```bash
python eval_video_qwen_mvbench.py \
    --model_path ./checkpoints/your-model \
    --video_dir ./dataset/MVBench/video \
    --output_path ./results/mvbench_all_results.json
```

This will:
- Load all 20 MVBench subsets
- Evaluate the model on each subset
- Generate detailed accuracy metrics per subset
- Save both raw results and summary with accuracies

### Evaluate Specific Subset

```bash
python eval_video_qwen_mvbench.py \
    --model_path ./checkpoints/your-model \
    --subset "action_sequence" \
    --video_dir ./dataset/MVBench/video
```

### Distributed Evaluation (Multiple GPUs)

```bash
python eval_video_qwen_mvbench.py \
    --model_path ./checkpoints/your-model \
    --video_dir ./dataset/MVBench/video \
    --distributed
```

## Arguments

- `--model_path`: Path to the pretrained model checkpoint
- `--dataset_name`: Hugging Face dataset name (default: "OpenGVLab/MVBench")
- `--split`: Dataset split to evaluate on (test/val)
- `--subset`: Specific subset to evaluate (if None, evaluates all subsets)
- `--video_dir`: Directory containing MVBench videos
- `--output_path`: Output file for results (optional)
- `--distributed`: Enable distributed evaluation
- `--conv_template`: Conversation template to use (default: "llama3")
- `--device`: Device to use for inference (default: "cuda")

## Output Format

### Raw Results (`mvbench_all_output.json`)
```json
[
    {
        "task_type": "action_sequence",
        "subset": "action_sequence",
        "data_path": "/path/to/video.mp4",
        "question": "What action is performed in the video?",
        "options": ["A. Walking", "B. Running", "C. Jumping"],
        "answer": "B",
        "prediction": "B. Running",
        "video": "video_filename.mp4",
        "idx": "action_sequence_0"
    }
]
```

### Summary Results (`mvbench_all_summary.json`)
```json
{
    "overall_accuracy": 0.75,
    "subset_accuracies": {
        "action_sequence": {
            "accuracy": 0.80,
            "correct": 80,
            "total": 100
        },
        "action_prediction": {
            "accuracy": 0.70,
            "correct": 70,
            "total": 100
        }
    },
    "total_samples": 2000,
    "results": [...]
}
```

## Key Improvements

1. **Automatic Multi-Subset Loading**: Loads all 20 MVBench subsets automatically
2. **Per-Subset Metrics**: Calculates and reports accuracy for each subset separately
3. **Robust Error Handling**: Continues evaluation if some subsets fail to load
4. **Detailed Logging**: Provides comprehensive logging of loading and evaluation progress
5. **Summary Reports**: Generates both raw results and summary files with accuracy metrics

## Requirements

- `datasets` library for loading MVBench data
- `transformers` for model and tokenizer  
- `torch` for model inference
- `accelerate` for distributed evaluation (optional)

## Example Output

```
Loading 20 MVBench subset(s): ['action_sequence', 'moving_count', ...]
Loading subset: action_sequence
Loaded 100 samples from action_sequence
Loading subset: moving_count  
Loaded 150 samples from moving_count
...
Loaded MVBench dataset with 2000 total samples from 20 subsets

Subset 'action_sequence': 80/100 = 0.8000
Subset 'moving_count': 120/150 = 0.8000
...
Overall Accuracy: 1600/2000 = 0.8000

Evaluation completed. Results saved to ./output/mvbench_all_output.json
Summary with accuracy metrics saved to ./output/mvbench_all_summary.json
Final overall accuracy: 0.8000
```
