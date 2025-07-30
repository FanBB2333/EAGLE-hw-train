#!/usr/bin/env python3
"""
Test script for MMLU evaluation
"""

import sys
import os
sys.path.append('/home6/fzy/repos/EAGLE')

from eval_mmlu import format_mmlu_question, evaluate_mmlu_answer

def test_format_question():
    """Test the MMLU question formatting function"""
    question = "What is the capital of France?"
    choices = ["London", "Berlin", "Paris", "Madrid"]
    subject = "geography"
    
    formatted = format_mmlu_question(question, choices, subject)
    print("Formatted question:")
    print(formatted)
    print("\n" + "="*50 + "\n")

def test_evaluate_answer():
    """Test the MMLU answer evaluation function"""
    test_cases = [
        ("A", 0, True),   # Correct
        ("B", 0, False),  # Wrong
        ("C", 2, True),   # Correct
        ("D", 3, True),   # Correct
        ("A Paris", 2, False),  # Wrong answer
        ("The answer is C", 2, True),  # Contains correct answer
    ]
    
    print("Answer evaluation tests:")
    for prediction, target_index, expected in test_cases:
        result = evaluate_mmlu_answer(prediction, target_index)
        status = "✓" if result == expected else "✗"
        print(f"{status} Prediction: '{prediction}', Target: {target_index}, Expected: {expected}, Got: {result}")

if __name__ == "__main__":
    print("Testing MMLU evaluation functions...")
    print("\n")
    
    test_format_question()
    test_evaluate_answer()
    
    print("\nTest completed!")
