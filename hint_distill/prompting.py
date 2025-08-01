"""
Unified prompting utilities for hint distillation with structured output.
"""

from typing import List, Dict
from enum import Enum


class ProgrammingLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    CPP = "cpp"
    JAVA = "java"
    JAVASCRIPT = "javascript"


class LanguageUtils:
    """Utility functions for working with programming languages."""
    
    @staticmethod
    def get_file_extension(language: ProgrammingLanguage) -> str:
        """Get the file extension for a programming language."""
        extensions = {
            ProgrammingLanguage.PYTHON: ".py",
            ProgrammingLanguage.CPP: ".cpp",
            ProgrammingLanguage.JAVA: ".java",
            ProgrammingLanguage.JAVASCRIPT: ".js"
        }
        return extensions.get(language, ".txt")
    
    @staticmethod
    def get_compile_command(language: ProgrammingLanguage) -> list:
        """Get the compilation command for a programming language."""
        commands = {
            ProgrammingLanguage.PYTHON: [],  # Python doesn't need compilation
            ProgrammingLanguage.CPP: ["g++", "-o", "program"],
            ProgrammingLanguage.JAVA: ["javac"],
            ProgrammingLanguage.JAVASCRIPT: []  # JavaScript doesn't need compilation
        }
        return commands.get(language, [])
    
    @staticmethod
    def get_run_command(language: ProgrammingLanguage) -> list:
        """Get the run command for a programming language."""
        commands = {
            ProgrammingLanguage.PYTHON: ["python"],
            ProgrammingLanguage.CPP: ["./program"],
            ProgrammingLanguage.JAVA: ["java", "program"],
            ProgrammingLanguage.JAVASCRIPT: ["node"]
        }
        return commands.get(language, [])
    
    @staticmethod
    def wrap_code_for_exec(language: ProgrammingLanguage, code: str) -> str:
        """Wrap code for execution with proper main function and input/output handling."""
        if language == ProgrammingLanguage.PYTHON:
            # Properly indent the code to fit inside the main function
            indented_code = '\n'.join('    ' + line for line in code.split('\n'))
            return f'''import sys

def main():
    # Read input from stdin
    data = sys.stdin.read().strip()
    if not data:
        return
    
    # Your solution code here
{indented_code}
    
    # Call your solution function with the input
    # You should implement the actual function call based on the problem
    result = solve(data)
    print(result)

if __name__ == "__main__":
    main()'''
        elif language == ProgrammingLanguage.CPP:
            return f'''#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

{code}

int main() {{
    // Read input from stdin
    string input;
    getline(cin, input);
    
    // Your solution function call here
    // You should implement the actual function call based on the problem
    int result = solve(input);
    cout << result << endl;
    
    return 0;
}}'''
        elif language == ProgrammingLanguage.JAVA:
            return f'''import java.util.*;

public class Solution {{
    {code}
    
    public static void main(String[] args) {{
        Scanner scanner = new Scanner(System.in);
        String input = scanner.nextLine();
        
        // Your solution function call here
        // You should implement the actual function call based on the problem
        int result = solve(input);
        System.out.println(result);
        
        scanner.close();
    }}
}}'''
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return f'''const readline = require('readline');

const rl = readline.createInterface({{
    input: process.stdin,
    output: process.stdout
}});

{code}

rl.question('', (input) => {{
    // Your solution function call here
    // You should implement the actual function call based on the problem
    const result = solve(input.trim());
    console.log(result);
    
    rl.close();
}});'''
        else:
            return code
    
    @staticmethod
    def get_code_template(language: ProgrammingLanguage) -> str:
        """Get the code template for a programming language."""
        templates = {
            ProgrammingLanguage.PYTHON: """# Your solution function here
def solve(input_data):
    # Process the input data and return the result
    # The input is a string that you need to parse based on the problem
    # Return the result as an integer or string as required
    
    # Example: if the problem asks for reading an integer and returning its square
    # num = int(input_data)
    # return num * num
    
    # Replace this with your actual solution
    result = 0
    return result""",
            
            ProgrammingLanguage.CPP: """// Your solution function here
int solve(std::string input) {
    // Process the input data and return the result
    // The input is a string that you need to parse based on the problem
    // Return the result as an integer
    
    // Example: if the problem asks for reading an integer and returning its square
    // int num = std::stoi(input);
    // return num * num;
    
    // Replace this with your actual solution
    int result = 0;
    return result;
}""",
            
            ProgrammingLanguage.JAVA: """// Your solution function here
public static int solve(String input) {
    // Process the input data and return the result
    // The input is a string that you need to parse based on the problem
    // Return the result as an integer
    
    // Example: if the problem asks for reading an integer and returning its square
    // int num = Integer.parseInt(input);
    // return num * num;
    
    // Replace this with your actual solution
    int result = 0;
    return result;
}""",
            
            ProgrammingLanguage.JAVASCRIPT: """// Your solution function here
function solve(input) {
    // Process the input data and return the result
    // The input is a string that you need to parse based on the problem
    // Return the result as an integer or string as required
    
    // Example: if the problem asks for reading an integer and returning its square
    // const num = parseInt(input);
    // return num * num;
    
    // Replace this with your actual solution
    let result = 0;
    return result;
}"""
        }
        return templates.get(language, templates[ProgrammingLanguage.PYTHON])


class CodeExtractor:
    """Utilities for extracting code from structured model output."""
    
    @staticmethod
    def extract_code_from_structured_output(output: str, language: ProgrammingLanguage) -> str:
        """Extract code from structured model output."""
        import re
        
        # Define patterns for different languages
        patterns = {
            ProgrammingLanguage.PYTHON: r'```python\n(.*?)\n```',
            ProgrammingLanguage.CPP: r'```cpp\n(.*?)\n```',
            ProgrammingLanguage.JAVA: r'```java\n(.*?)\n```',
            ProgrammingLanguage.JAVASCRIPT: r'```javascript\n(.*?)\n```'
        }
        
        pattern = patterns.get(language, r'```.*?\n(.*?)\n```')
        match = re.search(pattern, output, re.DOTALL)
        
        if match:
            code_content = match.group(1).strip()
            
            # Remove common non-code prefixes that the model might generate
            lines = code_content.split('\n')
            clean_lines = []
            for line in lines:
                # Skip common non-code lines
                if line.strip().lower() in ['code:', 'your code here:', 'solution:', '```']:
                    continue
                clean_lines.append(line)
            
            # If we have valid code lines, return them with language-specific fixes
            if clean_lines and any(line.strip() for line in clean_lines):
                raw_code = '\n'.join(clean_lines)
                # Apply language-specific fixes
                return CodeExtractor._fix_syntax_errors(raw_code, language)
        
        # Fallback: try to find any code block
        fallback_match = re.search(r'```(.*?)\n(.*?)\n```', output, re.DOTALL)
        if fallback_match:
            code_content = fallback_match.group(2).strip()
            
            # Clean up the code content
            lines = code_content.split('\n')
            clean_lines = []
            for line in lines:
                if line.strip().lower() in ['code:', 'your code here:', 'solution:', '```']:
                    continue
                clean_lines.append(line)
            
            if clean_lines and any(line.strip() for line in clean_lines):
                raw_code = '\n'.join(clean_lines)
                # Apply language-specific fixes
                return CodeExtractor._fix_syntax_errors(raw_code, language)
        
        # If no proper code blocks found, look for function definitions
        if language == ProgrammingLanguage.PYTHON:
            func_match = re.search(r'def\s+\w+\s*\(.*?\):(?:.*\n)*?.*?(?=\n\ndef|\n\n|\Z)', output, re.DOTALL)
            if func_match:
                raw_code = func_match.group(0).strip()
                # Apply Python-specific fixes
                return CodeExtractor._fix_syntax_errors(raw_code, language)
        
        # If still no code found, return empty string
        return ""
    
    @staticmethod
    def extract_plan_from_structured_output(output: str) -> str:
        """Extract approach/plan from structured model output."""
        import re
        
        # Look for approach section
        approach_match = re.search(r'Approach:\s*(.*?)(?=```|$)', output, re.DOTALL)
        if approach_match:
            approach = approach_match.group(1).strip()
            # Remove common prefixes and clean up
            approach = re.sub(r'^\[?\s*your one-sentence approach here\s*\]?\s*', '', approach, flags=re.IGNORECASE)
            return approach
        
        # Fallback: look for any text before code block
        code_match = re.search(r'```', output)
        if code_match:
            before_code = output[:code_match.start()].strip()
            # Remove common headers and instructions
            lines = [line.strip() for line in before_code.split('\n') if line.strip()]
            if lines:
                return lines[-1]  # Return the last line before code
        
        return ""
    
    @staticmethod
    def _fix_syntax_errors(code: str, language: ProgrammingLanguage) -> str:
        """Apply language-specific syntax fixes to common model generation errors."""
        if language == ProgrammingLanguage.PYTHON:
            return CodeExtractor._fix_python_syntax(code)
        elif language == ProgrammingLanguage.CPP:
            return CodeExtractor._fix_cpp_syntax(code)
        elif language == ProgrammingLanguage.JAVA:
            return CodeExtractor._fix_java_syntax(code)
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return CodeExtractor._fix_javascript_syntax(code)
        else:
            return code
    
    @staticmethod
    def _fix_python_syntax(code: str) -> str:
        """Fix common Python syntax errors in model-generated code."""
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        n = len(lines)
        
        while i < n:
            line = lines[i]
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                fixed_lines.append(line)
                i += 1
                continue
            
            # Fix: Missing indented block after function definition
            if (stripped_line.startswith('def ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('class ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('if ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('elif ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('else:') or (stripped_line.startswith('else ') and stripped_line.endswith(':'))) or \
               (stripped_line.startswith('for ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('while ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('try:') or (stripped_line.startswith('try ') and stripped_line.endswith(':'))) or \
               (stripped_line.startswith('except ') and stripped_line.endswith(':')) or \
               (stripped_line.startswith('finally:') or (stripped_line.startswith('finally ') and stripped_line.endswith(':'))) or \
               (stripped_line.startswith('with ') and stripped_line.endswith(':')):
                
                # Get the indentation level of the current line
                leading_spaces = len(line) - len(line.lstrip())
                indent = ' ' * (leading_spaces + 4)  # Add 4 spaces for the block
                
                # Check if the next line exists and is properly indented
                if i + 1 < n:
                    next_line = lines[i + 1]
                    next_stripped = next_line.strip()
                    
                    if next_stripped and not (next_line.startswith(indent) or next_line.startswith(' ' * (leading_spaces + 1))):
                        # The next line is not properly indented, so we need to fix it
                        fixed_lines.append(line)
                        
                        # Process all subsequent lines that should be indented but aren't
                        j = i + 1
                        has_content = False
                        
                        while j < n:
                            next_line_content = lines[j]
                            next_stripped_content = next_line_content.strip()
                            
                            if not next_stripped_content:
                                # Empty line, skip it for now
                                j += 1
                                continue
                            
                            # Check if this line represents the end of the current block
                            if next_line_content.startswith('def ') or \
                               next_line_content.startswith('class ') or \
                               next_line_content.startswith('if ') or \
                               next_line_content.startswith('elif ') or \
                               (next_line_content.startswith('else') and next_stripped_content.endswith(':')) or \
                               next_line_content.startswith('for ') or \
                               next_line_content.startswith('while ') or \
                               (next_line_content.startswith('with ') or \
                               next_line_content.startswith('try ') or \
                               next_line_content.startswith('except ') or \
                               next_line_content.startswith('finally')):
                                # We've reached a new block, so this one needs a pass if no content
                                if not has_content:
                                    fixed_lines.append(indent + 'pass')
                                break
                            
                            # Add the indented line
                            fixed_lines.append(indent + next_stripped_content)
                            has_content = True
                            j += 1
                        
                        # If we reached the end without adding content, add pass
                        if j >= n and not has_content:
                            fixed_lines.append(indent + 'pass')
                        
                        i = j - 1  # Continue from where we left off
                    else:
                        # Next line is properly indented, no need to fix
                        fixed_lines.append(line)
                else:
                    # This is the last line and it needs a body
                    fixed_lines.append(line)
                    fixed_lines.append(indent + 'pass')
            
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines)
    
    @staticmethod
    def _fix_cpp_syntax(code: str) -> str:
        """Fix common C++ syntax errors in model-generated code."""
        # For now, return as-is, but could add C++-specific fixes here
        return code
    
    @staticmethod
    def _fix_java_syntax(code: str) -> str:
        """Fix common Java syntax errors in model-generated code."""
        # For now, return as-is, but could add Java-specific fixes here
        return code
    
    @staticmethod
    def _fix_javascript_syntax(code: str) -> str:
        """Fix common JavaScript syntax errors in model-generated code."""
        # For now, return as-is, but could add JavaScript-specific fixes here
        return code


class PromptTemplates:
    """Unified prompt templates for training and evaluation."""
    
    @staticmethod
    def get_prompt_template(language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Get the structured prompt template for a given language."""
        language_names = {
            ProgrammingLanguage.PYTHON: "Python",
            ProgrammingLanguage.CPP: "C++",
            ProgrammingLanguage.JAVA: "Java",
            ProgrammingLanguage.JAVASCRIPT: "JavaScript"
        }
        
        code_block_markers = {
            ProgrammingLanguage.PYTHON: "```python",
            ProgrammingLanguage.CPP: "```cpp",
            ProgrammingLanguage.JAVA: "```java",
            ProgrammingLanguage.JAVASCRIPT: "```javascript"
        }
        
        language_name = language_names[language]
        code_marker = code_block_markers[language]
        
        # Get the code template for this language and include it in the prompt
        code_template = LanguageUtils.get_code_template(language)
        
        template_parts = [
            f"Solve this programming problem in {language_name}:",
            "",
            "Problem:",
            "{problem}",
            "",
            f"Write a {language_name} function that solves this problem.",
            f"Your code should read input from stdin and write output to stdout.",
            "",
            f"{code_marker}",
            code_template,
            f"{code_marker}"
        ]
        
        return "\n".join(template_parts)
    
    @staticmethod
    def get_example_prompt(problem, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Generate an example prompt with the given problem."""
        template = PromptTemplates.get_prompt_template(language)
        
        # Handle different input types
        if isinstance(problem, str):
            problem_text = problem
        elif isinstance(problem, dict):
            problem_text = problem.get("prompt", problem.get("question", str(problem)))
        else:
            problem_text = str(problem)
        
        # Try both {{problem}} and {problem} placeholders
        if "{{problem}}" in template:
            return template.replace("{{problem}}", problem_text)
        elif "{problem}" in template:
            return template.replace("{problem}", problem_text)
        else:
            return template + "\n\n# Problem:\n" + problem_text
    
    @staticmethod
    def get_dataset_solution_hint_prompt(problem: str, solution_code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Generate prompt for creating hints based on dataset solution."""
        language_name = {
            ProgrammingLanguage.PYTHON: "Python",
            ProgrammingLanguage.CPP: "C++", 
            ProgrammingLanguage.JAVA: "Java",
            ProgrammingLanguage.JAVASCRIPT: "JavaScript"
        }[language]
        
        return f"""Analyze this programming problem and its correct {language_name} solution to generate a concise and subtle hint.

Problem:
```{problem}```

Correct Solution:
```{language_name.lower()}
{solution_code}
```
Example of good subtle hint (do not copy this!):
Hint: If two adjacent values are equal, what impact does that have?

your hint must be very concise under 50 words and give the first step to the answer but do not give the answer. just subtle nudge in the right direction. Give only the hint text starting with "# Hint: " and nothing else."""
    
    @staticmethod
    def get_self_reflection_hint_prompt(problem: str, model_solution_plan: str, model_solution_code: str, correct_solution: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Generate prompt for a subtle hint based on comparing model and correct solutions, in the same style as other minimal hints."""

        language_name = {
            ProgrammingLanguage.PYTHON: "Python",
            ProgrammingLanguage.CPP: "C++",
            ProgrammingLanguage.JAVA: "Java", 
            ProgrammingLanguage.JAVASCRIPT: "JavaScript"
        }[language]
        
        return f"""Compare the model's attempted solution with the correct solution to generate a subtle, minimal hint. Do NOT reference what the model did wrong or reflect on the model's thinking. The hint should be phrased in exactly the same style as other subtle hints — like a breadcrumb that nudges the solver toward the right path.

    Problem:
    {problem}

    Model's Solution Plan:
    {model_solution_plan}

    Model's Solution Code:
    ```{language_name.lower()}
    {model_solution_code}
    ````

    Correct Solution:

    ```{language_name.lower()}
    {correct_solution}
    ```

    Your task is to analyze the difference and produce a **single minimal hint** that:

    1. Gives ONLY the first tiny step toward the insight (like a breadcrumb)
    2. NEVER states the algorithm, approach, or what’s missing
    3. May point out something to look at or ask a basic question
    4. Should be less than 50 words when possible

    Examples of good subtle hints:

    * "# Hint: What happens if you fix one side and vary the other?"
    * "# Hint: Try building a few small examples by hand"
    * "# Hint: What must be true about every valid output?"

    Return only the hint text starting with "# Hint: " and nothing else."""