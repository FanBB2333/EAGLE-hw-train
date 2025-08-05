#!/usr/bin/env python3
"""
Test script for the new video_metrics interface
"""
import sys
sys.path.append('.')

from fzy.video_metrics import evaluate_inference_results

def test_mvbench_interface():
    """Test MVBench evaluation interface"""
    # Mock MVBench inference results
    mvbench_results = [
        {
            "task": "mvbench",
            "prediction": "The best option is (D) The phone/camera.",
            "answer": "(D) The phone/camera.",
            "question": "Which object was taken by the person?",
        },
        {
            "task": "mvbench", 
            "prediction": "(A) The clothes.",
            "answer": "(B) The pillow.",
            "question": "What was picked up?",
        },
        {
            "task": "mvbench",
            "prediction": "The person took (C) The shoe.",
            "answer": "(C) The shoe.",
            "question": "What item was selected?",
        }
    ]
    
    print("Testing MVBench evaluation...")
    result = evaluate_inference_results(mvbench_results, "mvbench")
    print(f"MVBench Result: {result}")
    
    expected_accuracy = 2/3  # 2 correct out of 3
    actual_accuracy = result.get('accuracy', 0)
    print(f"Expected accuracy: {expected_accuracy:.4f}, Actual: {actual_accuracy:.4f}")
    return result

def test_temporal_localization_interface():
    """Test temporal localization evaluation interface"""
    # Mock ActivityNet inference results
    activitynet_results = [
        {
            "task": "activitynet",
            "prediction": "The event starts at 5 seconds.",
            "answer": [5.0, 15.0],  # [start, end]
            "question": "When does the action happen?",
        },
        {
            "task": "activitynet",
            "prediction": "It begins around 10 seconds in the video.",
            "answer": [8.0, 20.0],
            "question": "When does this occur?",
        },
        {
            "task": "activitynet",
            "prediction": "No clear timing information available.",
            "answer": [2.0, 8.0],
            "question": "When does the event take place?",
        }
    ]
    
    print("\nTesting ActivityNet evaluation...")
    result = evaluate_inference_results(activitynet_results, "activitynet")
    print(f"ActivityNet Result: {result}")
    return result

def test_unsupported_dataset():
    """Test unsupported dataset handling"""
    dummy_results = [{"prediction": "test", "answer": "test"}]
    
    print("\nTesting unsupported dataset...")
    result = evaluate_inference_results(dummy_results, "unsupported_dataset")
    print(f"Unsupported Dataset Result: {result}")
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Video Metrics Interface")
    print("=" * 50)
    
    try:
        # Test MVBench
        mvbench_result = test_mvbench_interface()
        
        # Test temporal localization
        temporal_result = test_temporal_localization_interface()
        
        # Test unsupported dataset
        unsupported_result = test_unsupported_dataset()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
