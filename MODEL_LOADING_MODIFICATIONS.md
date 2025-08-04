# Model Loading Modifications Summary

## Overview
Modified the EAGLE evaluation scripts to support dynamic model loading based on model type, specifically for qwen models.

## Changes Made

### 1. eval_video/eval_video_qwen.py
#### Extracted Model Loading Functions:
- **`load_video_model(args, modality='video')`**: 
  - Standalone function for loading video models
  - Handles tokenizer, model initialization, vision modules, and safetensor loading
  - Returns: `(model, tokenizer, image_processor, modality)`
  - Uses `torch.float16` and `model.cuda()`

- **`load_video_model_dist(args, accelerator, modality='video')`**:
  - Distributed version with Accelerator support
  - Similar functionality but adapted for distributed training
  - Uses `accelerator.prepare(model)` instead of `model.cuda()`

#### Updated Functions:
- **`evaluate_single_task()`**: Now calls `load_video_model()` instead of inline model loading
- **`evaluate_dist_single_task()`**: Now calls `load_video_model_dist()` instead of inline model loading

### 2. eval_acqa.py
#### Added Model Type Detection:
- **`is_qwen_model(model_path: str) -> bool`**: 
  - Detects if model is qwen-based by checking path for 'qwen' or 'qwen2vl'
  - Case-insensitive matching

#### Modified `evaluate()` Function:
```python
if is_qwen_model(args.model_path):
    # Use load_video_model for qwen models
    model, tokenizer, image_processor, modality = load_video_model(args, modality=modality)
    model_dtype = torch.float16  # qwen models use float16
else:
    # Use traditional load_pretrained_model for other models
    tokenizer, model, image_processor, max_length = load_pretrained_model(...)
    model_dtype = torch.bfloat16  # traditional models use bfloat16
```

#### Dynamic Data Type Handling:
- **Model dtype selection**: 
  - Qwen models: `torch.float16`
  - Traditional models: `torch.bfloat16`
- **Image tensor processing**: Uses `model_dtype` variable for proper type casting

## Benefits

### 1. Code Reusability
- Extracted model loading logic can be reused across different evaluation scripts
- Consistent model loading behavior across the codebase

### 2. Model Type Flexibility
- Automatic detection and appropriate loading method selection
- Support for both qwen and traditional models in the same script
- Fallback mechanism if qwen loading fails

### 3. Maintainability
- Centralized model loading logic reduces duplication
- Clear separation of concerns between model loading and evaluation
- Easy to extend for new model types

### 4. Data Type Consistency
- Proper data type handling based on model architecture
- Avoids type mismatch errors between model and input data

## Usage

### For qwen models:
```python
# Automatically detected and loaded with load_video_model()
args.model_path = "./checkpoints/finetune-video-llama3.2-3b-fzy-qwen2vl-llava-llava-294-168-old"
```

### For traditional models:
```python
# Automatically uses load_pretrained_model()
args.model_path = "./checkpoints/traditional-model-path"
```

## Import Requirements
The `eval_acqa.py` now requires access to `eval_video.eval_video_qwen.load_video_model`. If the import fails, it provides a clear error message and falls back to traditional loading methods.

## Backward Compatibility
- Traditional model loading paths remain unchanged
- Existing evaluation scripts continue to work without modification
- Graceful fallback mechanisms ensure robustness
