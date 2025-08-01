"""
Unified prompting utilities for hint distillation with structured output.
"""

from typing import List
from enum import Enum


class ProgrammingLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    CPP = "cpp"
    JAVA = "java"
    JAVASCRIPT = "javascript"


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
        
        return f"""Solve the following programming problem in {language_name}.

# Problem Description:
{{problem}}

# Instructions:
You must follow this exact format:

Approach: [One sentence describing your solution approach]

{code_marker}
[your {language_name} code here]
{code_marker}

Example:
Approach: Use a sliding window to find the longest substring with at most k distinct characters.

{code_marker}
def length_of_longest_substring_k_distinct(s, k):
    if k == 0:
        return 0
    
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        char = s[right]
        char_count[char] = char_count.get(char, 0) + 1
        
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length
{code_marker}

Now solve the given problem:

Approach: """
    
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
    def get_self_reflection_hint_prompt(problem: str, model_solution_plan: str, model_solution_code: str, correct_solution: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Get prompt for generating self-reflection hints by comparing model's solution to correct solution."""
        language_name = {
            ProgrammingLanguage.PYTHON: "Python",
            ProgrammingLanguage.CPP: "C++",
            ProgrammingLanguage.JAVA: "Java",
            ProgrammingLanguage.JAVASCRIPT: "JavaScript"
        }[language]
        
        return f"""You attempted to solve this programming problem and now need to reflect on what was challenging.

Problem:
{problem}

Your Attempted Solution Plan:
{model_solution_plan}

Your Attempted Solution Code:
```{language.value}
{model_solution_code}
```

The Correct Solution:
```{language.value}
{correct_solution}
```

Compare your solution to the correct solution and identify what was most challenging or where you struggled. Provide a helpful hint (1-2 sentences) that would guide someone through the difficult part without giving away the full solution.

Focus on:
- What concept was hardest to understand or implement correctly?
- Where did your approach differ from the optimal solution?
- What would have helped you arrive at the correct solution faster?

Hint: """

    @staticmethod
    def get_dataset_solution_hint_prompt(problem: str, solution_code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Get prompt for generating hints directly from dataset solution."""
        language_name = {
            ProgrammingLanguage.PYTHON: "Python",
            ProgrammingLanguage.CPP: "C++",
            ProgrammingLanguage.JAVA: "Java",
            ProgrammingLanguage.JAVASCRIPT: "JavaScript"
        }[language]
        
        return f"""Given this programming problem and its solution, generate a helpful hint that would guide someone to solve it without giving away the complete solution.

Problem:
{problem}

Solution Code:
```{language.value}
{solution_code}
```

Analyze the solution and identify the single most challenging or tricky part that someone trying to solve this problem might struggle with. Then, provide a helpful, specific hint (1-2 sentences) that would guide them through that challenging part.

Consider:
- What algorithmic concept is most complex?
- What edge case is hardest to handle?
- What implementation detail requires the most careful thought?
- What common mistake would someone make solving this?

The hint should be helpful but not give away the complete solution approach.

Hint: """


class CodeExtractor:
    """Extract code from structured model outputs."""
    
    @staticmethod
    def extract_code_from_structured_output(output: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> str:
        """Extract clean code from structured model output containing plans and code blocks."""
        code_markers = {
            ProgrammingLanguage.PYTHON: ["```python", "```"],
            ProgrammingLanguage.CPP: ["```cpp", "```"],
            ProgrammingLanguage.JAVA: ["```java", "```"],
            ProgrammingLanguage.JAVASCRIPT: ["```javascript", "```"]
        }
        
        start_marker, end_marker = code_markers[language]
        
        # Find the code block
        start_idx = output.find(start_marker)
        if start_idx == -1:
            # Try alternative markers or fallback
            start_idx = output.find("```")
            if start_idx == -1:
                # No code block found, return raw output
                return output.strip()
            start_marker = "```"
        
        end_idx = output.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            # No end marker found, return everything after start
            return output[start_idx + len(start_marker):].strip()
        
        # Extract the code
        code = output[start_idx + len(start_marker):end_idx].strip()
        return code
    
    @staticmethod
    def extract_plan_from_structured_output(output: str) -> str:
        """Extract the planning section from structured model output."""
        # Look for approach/planning sections
        lines = output.split('\n')
        plan_lines = []
        in_plan_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check for section headers
            if any(keyword in line_lower for keyword in ['approach', 'planning', 'plan:', '1. **approach']):
                in_plan_section = True
                if line.strip().startswith('#') or '**' in line:
                    continue  # Skip the header itself
            elif '```' in line_lower:
                in_plan_section = False
            elif line.strip().startswith('2. **solution code:'):
                in_plan_section = False
            
            if in_plan_section and line.strip():
                plan_lines.append(line)
        
        return '\n'.join(plan_lines).strip()


def code_block_marker(text: str) -> bool:
    """Check if text contains a code block marker."""
    return '```' in text


class LanguageUtils:
    """Language-specific utilities."""
    
    @staticmethod
    def get_file_extension(language: ProgrammingLanguage) -> str:
        """Get file extension for programming language."""
        extensions = {
            ProgrammingLanguage.PYTHON: ".py",
            ProgrammingLanguage.CPP: ".cpp",
            ProgrammingLanguage.JAVA: ".java",
            ProgrammingLanguage.JAVASCRIPT: ".js"
        }
        return extensions.get(language, ".txt")
    
    @staticmethod
    def get_execution_command(language: ProgrammingLanguage) -> List[str]:
        """Get command for executing code in a language."""
        import sys
        
        commands = {
            ProgrammingLanguage.PYTHON: [sys.executable],
            ProgrammingLanguage.CPP: ["g++", "-o", "temp_executable", "temp_file.cpp", "&&", "./temp_executable"],
            ProgrammingLanguage.JAVA: ["javac", "temp_file.java", "&&", "java", "temp_file"],
            ProgrammingLanguage.JAVASCRIPT: ["node"]
        }
        return commands.get(language, [sys.executable])
    
    @staticmethod
    def wrap_code_for_exec(language: ProgrammingLanguage, code: str) -> str:
        """Wrap code for execution with proper main function."""
        if language == ProgrammingLanguage.PYTHON:
            return code
        elif language == ProgrammingLanguage.CPP:
            return f'''#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

{code}

int main() {{
    // Main execution logic here
    // You should implement the actual function call based on the problem
    return 0;
}}'''
        elif language == ProgrammingLanguage.JAVA:
            return f'''public class Solution {{
{code}
    
    public static void main(String[] args) {{
        // Main execution logic here
        // You should implement the actual function call based on the problem
    }}
}}'''
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return f'''// Main execution
{code}

// Main function to run the solution
function main() {{
    // You should implement the actual function call based on the problem
}}

// Run the main function if this script is executed directly
if (require.main === module) {{
    main();
}}'''
        else:
            return code