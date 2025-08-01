"""
Utility functions for hint distillation.
"""

import os
import subprocess
import sys
import tempfile
import json
import shutil
from torch.utils.data import default_collate
import torch
from hint_distill.prompting import ProgrammingLanguage, LanguageUtils, PromptTemplates


def run_python(code_str: str, inp: str, timeout_s: int = 2) -> str:
    """Execute code in a subprocess, feed input to stdin, return stdout."""
    print(f"inp: {inp}")
    # If inp is a list, join it into a string
    if isinstance(inp, list):
        inp = "\n".join(map(str, inp))
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str)
        tmp_path = tmp.name
    cmd = [sys.executable, tmp_path]
    try:
        proc = subprocess.run(
            cmd,
            input=inp,
            capture_output=True,
            timeout=timeout_s,
            text=True
        )
        out = proc.stdout.strip()
        print(f"out: {out}")
    except subprocess.TimeoutExpired:
        out = "__TIMEOUT__"
    finally:
        os.remove(tmp_path)
    return out


def make_chunks(code: str, n_lines: int = 3):
    """Split code into chunks for training."""
    lines = code.splitlines()
    for i in range(0, len(lines), n_lines):
        yield lines[:i], lines[i : i + n_lines]


def auto_hint(ctx_lines, tgt_lines):
    """Generate automatic hint based on target code."""
    stripped = [l.strip() for l in tgt_lines if l.strip()]
    return f"# Hint: {stripped[0][:60]}" if stripped else "# Hint: next steps"


def run_code_multi_lang(code_str: str, inp: str, timeout_s: int = 2, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
    """Execute code in a subprocess for multiple programming languages."""
    print(f"inp: {inp}")
    
    # If inp is a list, join it into a string
    if isinstance(inp, list):
        inp = "\n".join(map(str, inp))
    
    # Get language-specific settings
    file_extension = LanguageUtils.get_file_extension(language)
    executed_code = LanguageUtils.wrap_code_for_exec(language, code_str)
    
    # Create temporary files with proper extension
    temp_dir = tempfile.mkdtemp()
    file_base = "temp_solution"
    file_name = file_base + file_extension
    file_path = os.path.join(temp_dir, file_name)
    
    try:
        # Write the code to file
        with open(file_path, 'w') as f:
            f.write(executed_code)
        
        # Build execution command based on language
        if language == ProgrammingLanguage.PYTHON:
            cmd = [sys.executable, file_path]
        elif language == ProgrammingLanguage.CPP:
            # Compile C++ first
            exe_path = os.path.join(temp_dir, file_base)
            compile_cmd = ["g++", "-o", exe_path, file_path]
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=30,
                text=True
            )
            if compile_result.returncode != 0:
                return f"COMPILATION_ERROR: {compile_result.stderr}"
            cmd = [exe_path]
        elif language == ProgrammingLanguage.JAVA:
            # Compile Java first
            compile_cmd = ["javac", file_path]
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                timeout=30,
                text=True,
                cwd=temp_dir
            )
            if compile_result.returncode != 0:
                return f"COMPILATION_ERROR: {compile_result.stderr}"
            cmd = ["java", file_base]
        elif language == ProgrammingLanguage.JAVASCRIPT:
            cmd = ["node", file_path]
        else:
            # Default to Python
            cmd = [sys.executable, file_path]
        
        # Set environment for subprocess
        env = os.environ.copy()
        if language == ProgrammingLanguage.JAVA:
            env["CLASSPATH"] = temp_dir
        
        # Execute the code
        proc = subprocess.run(
            cmd,
            input=inp,
            capture_output=True,
            timeout=timeout_s,
            text=True,
            cwd=temp_dir,
            env=env
        )
        
        out = proc.stdout.strip()
        if proc.returncode != 0:
            error_out = proc.stderr.strip()
            if error_out:
                out += f"\nERROR: {error_out}"
        
        print(f"out: {out}")
        
    except subprocess.TimeoutExpired:
        out = "__TIMEOUT__"
    except Exception as e:
        out = f"EXECUTION_ERROR: {str(e)}"
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return out


def generate_hint(problem: str, solution_plan: str, solution_code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, method: str = "self_reflection", model_solution_code: str = None):
    """Generate hint using specified method."""
    if method == "self_reflection":
        return generate_self_reflection_hint(problem, solution_plan, solution_code, language, model_solution_code)
    elif method == "dataset_solution":
        return generate_dataset_solution_hint(problem, solution_code, language)
    else:
        raise ValueError(f"Unknown hint generation method: {method}")

def generate_self_reflection_hint(problem: str, model_solution_plan: str, model_solution_code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, correct_solution: str = None):
    """Generate intelligent hint based on self-reflection comparing model solution to correct solution."""
    try:
        if correct_solution is None:
            # If no correct solution provided, analyze the model's own solution
            hint_prompt = PromptTemplates.get_dataset_solution_hint_prompt(problem, model_solution_code, language)
            return _analyze_solution_complexity(model_solution_code, language)
        else:
            # Create the self-reflection prompt with comparison
            hint_prompt = PromptTemplates.get_self_reflection_hint_prompt(problem, model_solution_plan, model_solution_code, correct_solution, language)
            # For now, analyze the difference between solutions or fall back to solution complexity
            return _analyze_solution_difference(problem, model_solution_code, correct_solution, language)
    except Exception as e:
        # Fallback to analyzing the model's solution
        return _analyze_solution_complexity(model_solution_code, language)

def generate_dataset_solution_hint(problem: str, solution_code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON):
    """Generate intelligent hint based directly on dataset solution."""
    try:
        hint_prompt = PromptTemplates.get_dataset_solution_hint_prompt(problem, solution_code, language)
        # For now, return a simple hint based on code analysis
        hint = _analyze_solution_complexity(solution_code, language)
        return hint
    except Exception as e:
        # Fallback to simple hint generation
        return _generate_simple_hint(solution_code, language)

def _analyze_solution_difference(problem: str, model_solution: str, correct_solution: str, language: ProgrammingLanguage) -> str:
    """Analyze differences between model solution and correct solution to identify hint-worthy areas."""
    # For now, fall back to analyzing the correct solution's complexity
    # In a full implementation, this would compare the two solutions
    return _analyze_solution_complexity(correct_solution, language)


def _analyze_solution_complexity(solution_code: str, language: ProgrammingLanguage) -> str:
    """Analyze solution code to identify challenging parts and generate hints."""
    lines = solution_code.split('\n')
    code_lower = solution_code.lower()
    
    # Look for indicators of complexity in the code - more comprehensive patterns
    complexity_patterns = [
        # Specific recursive patterns
        (r'\brecursive\b', "Recursion can be tricky - make sure to define your base case and recursive step clearly"),
        (r'def \w+\([^)]*\):\s*.*\bdef \w+\([^)]*\)', "Recursion detected - ensure you have proper base cases to prevent infinite recursion"),
        (r'return \w+\([^)]*\)\s*.*\bn[-+]1\b', "Recursive call detected - focus on the base case and how the problem size reduces"),
        
        # Nested loops patterns
        (r'for.*in.*:\s*.*for.*in.*:', "Nested loops detected - this often leads to O(n²) complexity, consider if optimization is possible"),
        (r'for.*in.*:\s*.*while.*:', "Nested loop with while - be careful about loop conditions and termination"),
        
        # Dynamic programming patterns
        (r'\bmemo\b|\bdp\b|cache', "Dynamic programming detected - identify overlapping subproblems and optimal substructure"),
        (r'[a-zA-Z_]\[n\].*=.*[a-zA-Z_]\[n[-+]', "Memoization pattern - ensure you're storing and reusing computed results efficiently"),
        
        # Complex conditionals
        (r'if.*:\s*.*if.*:\s*.*else.*:', "Complex nested conditionals - consider simplifying logic or using helper functions"),
        (r'elif.*:\s*.*elif.*:', "Multiple else-if branches - consider if a lookup table or switch would be cleaner"),
        
        # Edge case handling
        (r'if.*<=.*<=.*:', "Range boundary check - ensure your conditions handle all edge cases correctly"),
        (r'if.*==.*0.*:\s*.*if.*==.*1.*:', "Multiple equality checks - consider combining logic or using data structures"),
        
        # String manipulation complexity
        (r'\.split\(\).*\.join\(', "String splitting and joining - watch out for empty strings and whitespace handling"),
        (r'\.replace\(.*\).*\.replace\(', "Multiple string replacements - consider if regex or a single pass would be more efficient"),
        (r'\[.*:-1.*\]|\[::-1\]', "String slicing/reversal - pay attention to off-by-one errors and inclusive/exclusive bounds"),
        
        # List comprehensions
        (r'\[.*for.*in.*if.*for.*in', "Nested list comprehension - these can be hard to read and debug"),
        (r'\[.*for.*in.*for.*in', "Multiple for loops in comprehension - consider breaking into nested loops for clarity"),
        
        # Exception handling
        (r'try:\s*.*except.*:\s*.*finally:', "Complex exception handling - ensure resources are properly managed"),
        (r'except.*Exception.*:', "Broad exception handling - consider catching specific exceptions for better debugging"),
        
        # Function complexity
        (r'def.*:\s*.*return.*return', "Multiple return statements - ensure they cover all code paths"),
        (r'def.*:\s*.*if.*return.*else.*return', "Conditional returns - verify that both paths are correct"),
        
        # Array/List operations
        (r'\.append\(.*\).*\.pop\(', "List append and pop operations - be mindful of time complexity and list bounds"),
        (r'sort\(\).*sorted\(', "Multiple sorting operations - consider if sorting only once would be more efficient"),
    ]
    
    # Check patterns in order of specificity
    import re
    for pattern, hint in complexity_patterns:
        if re.search(pattern, code_lower):
            return f"# Hint: {hint}"
    
    # Fallback to general analysis based on code structure
    function_count = solution_code.count('def ')
    loop_count = solution_code.lower().count('for ') + solution_code.lower().count('while ')
    
    if function_count >= 3:
        return "# Hint: Multiple functions require careful design - ensure proper interfaces and data flow between them"
    elif loop_count >= 2:
        return "# Hint: Multiple loops suggest important iteration logic - focus on loop invariants and termination conditions"
    elif 'if ' in solution_code and len([line for line in lines if 'if ' in line.lower()]) >= 3:
        return "# Hint: Multiple conditional branches - ensure all cases are covered and logic is sound"
    else:
        return "# Hint: Focus on the core algorithm logic and verify it handles all edge cases correctly"


def _generate_simple_hint(solution_code: str, language: ProgrammingLanguage) -> str:
    """Generate a simple hint based on code structure when self-reflection isn't available."""
    lines = [l.strip() for l in solution_code.split('\n') if l.strip()]
    
    if lines:
        # Look for key patterns in the first few meaningful lines
        for line in lines[:5]:
            if any(keyword in line.lower() for keyword in ['if ', 'for ', 'while ', 'def ', 'class ']):
                hint_text = line[:50]
                return f"# Hint: Focus on implementing {hint_text}"
    
    return "# Hint: Break down the problem into smaller, manageable components"


def hint_distill_collate_fn(batch):
    """Custom collate function that preserves hint distillation keys."""
    # Standard collation for regular keys
    standard_batch = default_collate(batch)
    
    # Manually add the custom hint distillation keys
    for key in ["input_ids_no_hint", "input_ids_with_hint", "labels_no_hint", "labels_with_hint"]:
        if key in batch[0]:
            standard_batch[key] = torch.stack([item[key] for item in batch])
    
    return standard_batch