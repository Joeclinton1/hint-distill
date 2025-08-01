"""
Data utilities and dataset classes for hint distillation.
"""

import json
from typing import Dict, Any, Optional, List
import torch
from datasets import load_dataset
from hint_distill.utils import generate_self_reflection_hint, make_chunks, auto_hint, generate_hint
from hint_distill.prompting import ProgrammingLanguage, PromptTemplates
from hint_distill.evaluation import generate_code


class HintDataset(torch.utils.data.Dataset):
    """Dataset for hint distillation training with/without hints."""
    
    def __init__(self, tokenizer, code, n_lines=3, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON):
        self.samples = []
        for ctx, tgt in make_chunks(code, n_lines):
            ctx_src = "\n".join(ctx)
            tgt_src = "\n".join(tgt)
            hint = auto_hint(ctx, tgt)

            # Ensure we have some content
            if not ctx_src.strip() and not tgt_src.strip():
                continue
            
            # For empty context, add a minimal prompt
            if not ctx_src.strip():
                ctx_src = f"# Write the following {language.value} code:\n"
            
            with_hint = tokenizer(ctx_src + "\n" + hint, return_tensors="pt").input_ids[0]
            no_hint = tokenizer(ctx_src, return_tensors="pt").input_ids[0]
            tgt_ids = tokenizer(tgt_src, return_tensors="pt").input_ids[0]

            # Skip empty samples
            if len(no_hint) == 0 or len(with_hint) == 0 or len(tgt_ids) == 0:
                continue

            labels_with_hint = torch.full_like(with_hint, -100)
            if len(tgt_ids) <= len(with_hint):
                labels_with_hint[-len(tgt_ids):] = tgt_ids
            else:
                # Truncate tgt_ids if it's longer than the input
                labels_with_hint[-len(with_hint):] = tgt_ids[-len(with_hint):]

            labels_no_hint = torch.full_like(no_hint, -100)
            if len(tgt_ids) <= len(no_hint):
                labels_no_hint[-len(tgt_ids):] = tgt_ids
            else:
                # Truncate tgt_ids if it's longer than the input
                labels_no_hint[-len(no_hint):] = tgt_ids[-len(no_hint):]

            self.samples.append({
                "input_ids": no_hint,
                "labels": labels_no_hint,
                "input_ids_no_hint": no_hint,
                "input_ids_with_hint": with_hint,
                "labels_no_hint": labels_no_hint,
                "labels_with_hint": labels_with_hint,
                "language": language.value,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MultiLanguageHintDataset(torch.utils.data.Dataset):
    """Dataset for multi-language hint distillation with structured prompts."""
    
    def __init__(self, tokenizer, problems: List[Dict[str, Any]], n_lines=3):
        self.samples = []
        
        for problem in problems:
            code = problem.get("solution", "")
            question = problem.get("question", problem.get("prompt", ""))
            language_name = problem.get("language", problem.get("metadata", {}).get("language", "python"))
            
            # Map language name to ProgrammingLanguage enum
            try:
                language = ProgrammingLanguage(language_name.lower())
            except ValueError:
                language = ProgrammingLanguage.PYTHON  # Default to Python
            
            # Create structured prompt for this problem
            structured_prompt = PromptTemplates.get_example_prompt(question, language)
            
            # Split code into chunks for training
            for i, (ctx, tgt) in enumerate(make_chunks(code, n_lines)):
                if i >= 10:  # Limit chunks per problem to avoid large datasets
                    break
                    
                ctx_src = "\n".join(ctx)
                tgt_src = "\n".join(tgt)
                
                # Generate context-aware hint
                hint = self._generate_structured_hint(ctx_src, tgt_src, question, language)
                
                # Ensure we have content
                if not ctx_src.strip() and not tgt_src.strip():
                    continue
                
                if not ctx_src.strip():
                    ctx_src = f"# Continue writing the {language.value} solution:\n"
                
                # Create training samples with structured prompts
                with_hint_prompt = structured_prompt + f"\n{ctx_src}\n{hint}"
                no_hint_prompt = structured_prompt + f"\n{ctx_src}"
                
                # Tokenize
                with_hint_ids = tokenizer(with_hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                no_hint_ids = tokenizer(no_hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                tgt_ids = tokenizer(tgt_src, return_tensors="pt", truncation=True, max_length=256).input_ids[0]
                
                # Skip empty samples
                if len(no_hint_ids) == 0 or len(with_hint_ids) == 0 or len(tgt_ids) == 0:
                    continue
                
                # Create labels
                labels_with_hint = torch.full_like(with_hint_ids, -100)
                if len(tgt_ids) <= len(with_hint_ids):
                    labels_with_hint[-len(tgt_ids):] = tgt_ids
                
                labels_no_hint = torch.full_like(no_hint_ids, -100)
                if len(tgt_ids) <= len(no_hint_ids):
                    labels_no_hint[-len(tgt_ids):] = tgt_ids
                
                self.samples.append({
                    "input_ids": no_hint_ids,
                    "labels": labels_no_hint,
                    "input_ids_no_hint": no_hint_ids,
                    "input_ids_with_hint": with_hint_ids,
                    "labels_no_hint": labels_no_hint,
                    "labels_with_hint": labels_with_hint,
                    "language": language.value,
                    "problem_id": problem.get("metadata", {}).get("problem_id", f"unknown_{len(self.samples)}"),
                    "structured_prompt": structured_prompt,
                    "context": ctx_src,
                    "target": tgt_src,
                    "hint": hint
                })
    
    def _generate_structured_hint(self, ctx_src: str, tgt_src: str, question: str, language: ProgrammingLanguage) -> str:
        """Generate a structured hint based on context and question."""
        # Extract key parts of the target code
        tgt_lines = [line.strip() for line in tgt_src.split('\n') if line.strip()]
        
        if not tgt_lines:
            return f"# Hint: Complete the {language.value} solution"
        
        first_line = tgt_lines[0]
        
        # Language-specific hint generation
        if language == ProgrammingLanguage.PYTHON:
            if first_line.startswith('def '):
                return f"# Hint: Implement the function: {first_line}"
            elif first_line.startswith('class '):
                return f"# Hint: Define the class: {first_line}"
            elif any(op in first_line for op in ['for ', 'if ', 'while ']):
                return f"# Hint: Add {first_line}"
            else:
                return f"# Hint: {first_line[:50]}"
        
        elif language == ProgrammingLanguage.CPP:
            if first_line.startswith('int ') or first_line.startswith('void ') or first_line.startswith('char '):
                return f"# Hint: Implement the function: {first_line}"
            elif first_line.startswith('class '):
                return f"# Hint: Define the class: {first_line}"
            elif '{' in first_line:
                return f"# Hint: Add {first_line}"
            else:
                return f"# Hint: {first_line[:50]}"
        
        else:
            return f"# Hint: Implement: {first_line[:50]}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ProblemLoader:
    """Handles loading programming problems from various sources."""
    
    @staticmethod
    def load_from_apps(index: int, difficulty: str = "competition") -> Optional[Dict[str, Any]]:
        """Load a problem from the APPS dataset."""
        try:
            ds = load_dataset("codeparrot/apps", split="test[:100]", 
                            difficulties=[difficulty], trust_remote_code=True)
            ds_list = list(ds)
            
            if index >= len(ds_list):
                print(f"❌ Index {index} out of range. Dataset has {len(ds_list)} examples.")
                return None
                
            ex = ds_list[index]
            
            # Parse input_output and solutions safely
            input_output = ex.get("input_output", "{}")
            solutions_data = ex.get("solutions", "[]")
            
            try:
                tests = json.loads(input_output) if isinstance(input_output, str) else input_output
            except (json.JSONDecodeError, TypeError):
                tests = {"inputs": [], "outputs": []}
                
            try:
                solutions = json.loads(solutions_data) if isinstance(solutions_data, str) else solutions_data
            except (json.JSONDecodeError, TypeError):
                solutions = []
            
            problem = {
                "question": ex.get("question", ""),
                "solution": solutions[0] if solutions else "",
                "tests": tests,
                "metadata": {
                    "problem_id": ex.get("problem_id", f"unknown_{index}"),
                    "difficulty": ex.get("difficulty", difficulty),
                    "url": ex.get("url", ""),
                    "source": "apps"
                }
            }
            
            print(f"✅ Loaded APPS problem {problem['metadata']['problem_id']}")
            return problem
            
        except Exception as e:
            print(f"❌ Failed to load APPS problem: {e}")
            return None
    
    @staticmethod
    def load_from_json(json_path: str) -> Optional[Dict[str, Any]]:
        """Load a problem from JSON file."""
        try:
            with open(json_path, 'r') as f:
                problem = json.load(f)
            
            problem.setdefault("metadata", {})
            problem["metadata"]["source"] = "json_file"
            problem["metadata"]["json_path"] = json_path
            
            print(f"✅ Loaded problem from JSON")
            return problem
        except Exception as e:
            print(f"❌ Failed to load problem from JSON: {e}")
            return None


class FlexibleHintDataset(torch.utils.data.Dataset):
    """Dataset for hint distillation supporting both self-reflection and dataset solution methods."""
    
    def __init__(self, tokenizer, problems_data: List[Dict[str, Any]], model, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, 
                 hint_method: str = "self_reflection", max_problems: int = None, hint_log_file: str = None):
        self.samples = []
        self.hint_method = hint_method
        self.hint_log_file = hint_log_file
        
        problems_processed = 0
        print(f"DEBUG: FlexibleHintDataset initializing with {len(problems_data)} problems, hint_method={hint_method}")
        
        for i, problem_data in enumerate(problems_data):
            print(f"DEBUG: Processing problem {i}")
            if max_problems and problems_processed >= max_problems:
                break
            
            try:
                # Extract problem information
                problem = problem_data.get("prompt", problem_data.get("question", ""))
                correct_solution = problem_data.get("solution", problem_data.get("code", ""))
                
                if not problem.strip():
                    continue
                
                # Generate model's attempted solution (for self-reflection method)
                model_solution_plan = ""
                model_solution_code = ""
                
                if hint_method == "self_reflection":
                    # Generate model's attempt
                    print(f"DEBUG: Calling generate_code for problem {i}...")
                    generation_result = generate_code(
                        model, 
                        tokenizer, 
                        problem_data, 
                        language=language,
                        max_new_tokens=2048,
                        stream=False
                    )
                    print(f"DEBUG: generate_code completed for problem {i}")
                    
                    if "error" not in generation_result:
                        model_solution_plan = generation_result.get("plan", "")
                        model_solution_code = generation_result["code"]
                
                # Generate hint using the specified method
                if hint_method == "self_reflection":
                    hint = generate_hint(
                        problem, 
                        model_solution_plan, 
                        model_solution_code, 
                        language=language,
                        method="self_reflection",
                        model_solution_code=correct_solution if correct_solution else None
                    )
                elif hint_method == "dataset_solution":
                    solution_for_hint = correct_solution if correct_solution else model_solution_code
                    hint = generate_hint(
                        problem, 
                        model_solution_plan, 
                        solution_for_hint, 
                        language=language,
                        method="dataset_solution"
                    )
                else:
                    raise ValueError(f"Unknown hint method: {hint_method}")
                
                # Choose which solution to use as the target
                target_solution = correct_solution if correct_solution else model_solution_code
                
                if not target_solution.strip():
                    continue
                
                # Create structured prompts with and without hint
                no_hint_prompt = PromptTemplates.get_example_prompt(problem, language)
                hint_prompt = self._create_hint_prompt(problem, hint, language)
                
                # Tokenize
                no_hint_ids = tokenizer(no_hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                hint_ids = tokenizer(hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                solution_ids = tokenizer(target_solution, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                
                # Skip empty samples
                if len(no_hint_ids) == 0 or len(hint_ids) == 0 or len(solution_ids) == 0:
                    continue
                
                # Create labels
                labels_no_hint = torch.full_like(no_hint_ids, -100)
                if len(solution_ids) <= len(no_hint_ids):
                    labels_no_hint[-len(solution_ids):] = solution_ids
                
                labels_with_hint = torch.full_like(hint_ids, -100)
                if len(solution_ids) <= len(hint_ids):
                    labels_with_hint[-len(solution_ids):] = solution_ids
                
                self.samples.append({
                    "input_ids_no_hint": no_hint_ids,
                    "input_ids_with_hint": hint_ids,
                    "labels_no_hint": labels_no_hint,
                    "labels_with_hint": labels_with_hint,
                    "language": language.value,
                    "hint_method": hint_method,
                    "problem": problem, 
                    "model_solution_plan": model_solution_plan,
                    "model_solution_code": model_solution_code,
                    "correct_solution": correct_solution,
                    "target_solution": target_solution,
                    "hint": hint,
                    "metadata": problem_data.get("metadata", {})
                })
                
                # Log hint to file if hint_log_file is provided
                if self.hint_log_file:
                    # Import here to avoid circular dependency
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from hint_distill_demo import log_hint_to_file
                    log_hint_to_file(
                        self.hint_log_file,
                        problem_data.get("metadata", {}),
                        hint,
                        model_solution_plan[:200] + "..." if len(model_solution_plan) > 200 else model_solution_plan,
                        target_solution[:200] + "..." if len(target_solution) > 200 else target_solution,
                        hint_method
                    )
                
                problems_processed += 1
                
            except Exception as e:
                print(f"⚠️  Failed to process problem: {e}")
                continue
        
        print(f"✅ Generated {len(self.samples)} training samples using {hint_method} method")
    
    def _create_hint_prompt(self, problem: str, hint: str, language: ProgrammingLanguage) -> str:
        """Create a prompt that includes the problem and hint."""
        base_prompt = PromptTemplates.get_example_prompt(problem, language)
        return base_prompt + f"\n\n# Hint:\n{hint}"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SelfReflectionHintDataset(torch.utils.data.Dataset):
    """Backward compatibility wrapper for self-reflection dataset."""
    
    def __init__(self, tokenizer, solutions: List[Dict[str, str]], language: ProgrammingLanguage = ProgrammingLanguage.PYTHON):
        # Convert solutions format to problems format for new dataset
        problems_data = []
        for solution_data in solutions:
            problems_data.append({
                "prompt": solution_data.get("problem", ""),
                "solution": solution_data.get("code", ""),
                "metadata": solution_data.get("metadata", {})
            })
        
        # Use the new flexible dataset with self-reflection method
        self.flexible_dataset = FlexibleHintDataset(
            tokenizer, 
            problems_data, 
            model=None,  # Will be set in create_from_problems
            language=language,
            hint_method="self_reflection"
        )
    
    def __len__(self):
        return len(self.flexible_dataset)
    
    def __getitem__(self, idx):
        return self.flexible_dataset[idx]
    
    @staticmethod
    def create_from_problems(tokenizer, problems: List[Dict[str, Any]], model, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, max_problems: int = None):
        """Create dataset using self-reflection method."""
        dataset = FlexibleHintDataset(tokenizer, problems, model, language, "self_reflection", max_problems)
        # Return a wrapper to maintain backward compatibility
        return dataset