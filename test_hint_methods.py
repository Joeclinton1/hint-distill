#!/usr/bin/env python3
"""
Test both hint generation methods to ensure they work correctly.
"""

import sys
sys.path.append('.')

from hint_distill.prompting import ProgrammingLanguage
from hint_distill.utils import generate_hint

def test_both_hint_methods():
    """Test both self_reflection and dataset_solution hint methods."""
    print("Testing both hint generation methods...")
    
    # Test problem and solutions
    problem = "Write a function that calculates the nth Fibonacci number using dynamic programming."
    model_plan = "Use recursion with memoization to avoid recalculating Fibonacci numbers"
    model_code = """def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)"""
    
    correct_solution = """def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]"""
    
    # Test 1: Self-reflection method
    print("\n" + "="*60)
    print("1. Testing 'self_reflection' method:")
    print("="*60)
    
    hint1 = generate_hint(
        problem=problem,
        solution_plan=model_plan,
        solution_code=model_code,
        language=ProgrammingLanguage.PYTHON,
        method="self_reflection",
        model_solution_code=correct_solution
    )
    print(f"Self-reflection hint: {hint1}")
    
    # Test 2: Dataset solution method
    print("\n" + "="*60)
    print("2. Testing 'dataset_solution' method:")
    print("="*60)
    
    hint2 = generate_hint(
        problem=problem,
        solution_plan=model_plan,
        solution_code=correct_solution,
        language=ProgrammingLanguage.PYTHON,
        method="dataset_solution"
    )
    print(f"Dataset solution hint: {hint2}")
    
    # Test 3: Error handling
    print("\n" + "="*60)
    print("3. Testing error handling:")
    print("="*60)
    
    try:
        hint3 = generate_hint(
            problem=problem,
            solution_plan=model_plan,
            solution_code=model_code,
            language=ProgrammingLanguage.PYTHON,
            method="invalid_method"
        )
        print("ERROR: Should have raised ValueError for invalid method")
    except ValueError as e:
        print(f"✅ Correctly raised error: {e}")
    
    # Test 4: Different code complexities
    print("\n" + "="*60)
    print("4. Testing different code complexities:")
    print("="*60)
    
    test_cases = [
        {
            "name": "Nested loops",
            "problem": "Find all pairs in an array that sum to target",
            "solution": """def find_pairs(arr, target):
    pairs = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] == target:
                pairs.append((arr[i], arr[j]))
    return pairs"""
        },
        {
            "name": "Recursion",
            "problem": "Calculate factorial recursively",
            "solution": """def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)"""
        },
        {
            "name": "String manipulation",
            "problem": "Reverse words in a string",
            "solution": """def reverse_words(s):
    return ' '.join(s.split()[::-1])"""
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}:")
        
        # Test with dataset_solution method
        hint_ds = generate_hint(
            problem=test_case['problem'],
            solution_plan="",
            solution_code=test_case['solution'],
            language=ProgrammingLanguage.PYTHON,
            method="dataset_solution"
        )
        
        # Test with self_reflection method (without correct solution for simplicity)
        hint_sr = generate_hint(
            problem=test_case['problem'],
            solution_plan="",
            solution_code=test_case['solution'],
            language=ProgrammingLanguage.PYTHON,
            method="self_reflection"
        )
        
        print(f"  Dataset solution hint: {hint_ds}")
        print(f"  Self-reflection hint: {hint_sr}")
    
    print("\n" + "="*60)
    print("✅ All hint generation method tests completed!")
    print("="*60)
    
    print("\nUsage in training:")
    print("  --hint_method self_reflection    # Model attempts → compares → self-reflects")
    print("  --hint_method dataset_solution # Direct hint from provided solution")

if __name__ == "__main__":
    test_both_hint_methods()