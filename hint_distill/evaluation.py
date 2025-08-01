"""
Evaluation utilities for hint distillation.
"""

import time
from typing import Dict, Any, Optional
from threading import Thread
import torch
from transformers import AutoTokenizer, TextIteratorStreamer

# Apply torch optimizations for faster inference
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Try to enable flash attention if available
    try:
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
    except:
        pass  # Flash attention not available

from hint_distill.utils import run_code_multi_lang
from hint_distill.prompting import (
    PromptTemplates,
    CodeExtractor,
    ProgrammingLanguage
)


def tests_pass(code_str: str, tests, timeout=2, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON):
    """Check if code passes all test cases."""
    import json
    
    # APPS dataset format: tests is a dict with 'inputs' and 'outputs' lists
    if isinstance(tests, dict):
        inputs = tests.get("inputs", [])
        outputs = tests.get("outputs", [])
        for _in, _out in zip(inputs, outputs):
            # Process input as JSON if it's a list
            if isinstance(_in, list):
                input_str = json.dumps(_in)
            else:
                input_str = str(_in)
            
            pred = run_code_multi_lang(code_str, input_str, timeout, language)
            if pred.strip() != str(_out).strip():
                return False
        return True
    # Original format: list of dictionaries with 'input' and 'output' keys
    else:
        for io_pair in tests:
            if isinstance(io_pair, dict):
                _in, _out = io_pair.get("input"), io_pair.get("output")
            else:
                continue
            
            # Process input as JSON if it's a list
            if isinstance(_in, list):
                input_str = json.dumps(_in)
            else:
                input_str = str(_in)
            
            pred = run_code_multi_lang(code_str, input_str, timeout, language)
            if pred.strip() != str(_out).strip():
                return False
        return True


def pass_at_k(model, tokenizer, problem, language=ProgrammingLanguage.PYTHON, k=5, temp=0.2, max_new=256):
    """Calculate pass@k metric by generating k solutions and testing them."""
    ok = False
    
    structured_prompt = PromptTemplates.get_example_prompt(problem, language)
    
    for _ in range(k):
        inputs = tokenizer(structured_prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_output = tokenizer.decode(gen[0], skip_special_tokens=True)
        
        # Extract clean code from structured output
        if full_output.startswith(structured_prompt):
            full_output = full_output[len(structured_prompt):]
        
        code = CodeExtractor.extract_code_from_structured_output(full_output, language)
        
        if tests_pass(code, problem.get("tests", {}), language=language):
            ok = True
            break
    return 1.0 if ok else 0.0


def generate_code(
    model, 
    tokenizer: AutoTokenizer, 
    problem: Dict[str, Any],
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    max_new_tokens: int = 512,
    temperature: float = 0.2, 
    num_beams: int = 1, 
    do_sample: bool = True,
    extract_plan: bool = True,
    stream: bool = False
) -> Dict[str, Any]:
    """Generate code from model with structured prompting and code extraction."""
    start_time = time.time()
    
    try:
        # Create structured prompt
        structured_prompt = PromptTemplates.get_example_prompt(problem, language)
        
        # Enable faster inference mode
        model.eval()
        
        if stream and num_beams == 1 and do_sample:
            # Use streaming generation
            raw_output, full_output, extraction_time = _generate_code_streaming(
                model, tokenizer, structured_prompt, max_new_tokens, temperature, extract_plan
            )
            generation_time = time.time() - start_time - extraction_time
            generation_time += extraction_time  # Add extraction time back
        else:
            # Use non-streaming generation
            raw_output, full_output = _generate_code_non_streaming(
                model, tokenizer, structured_prompt, max_new_tokens, temperature, num_beams, do_sample
            )
            generation_time = time.time() - start_time
        
        # Extract clean code and plan from structured output
        extracted_code = CodeExtractor.extract_code_from_structured_output(full_output, language)
        extracted_plan = CodeExtractor.extract_plan_from_structured_output(full_output) if extract_plan else ""
        
        return {
            "code": extracted_code,
            "plan": extracted_plan,
            "full_output": full_output,
            "raw_output": raw_output,
            "structured_prompt": structured_prompt,
            "generation_time": generation_time,
            "tokens": len(full_output.split()),
            "lines": len(extracted_code.split('\n')) if extracted_code else 0,
            "language": language.value,
            "temperature": temperature,
            "do_sample": do_sample,
            "num_beams": num_beams
        }
        
    except Exception as e:
        return {
            "code": "",
            "plan": "",
            "full_output": "",
            "error": str(e)
        }


def _generate_code_streaming(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, extract_plan: bool):
    """Generate code with streaming output using TextIteratorStreamer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("   ├─ Generating response...")
    print("   │")
    
    # Set up the streamer
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60.0,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Set up generation arguments with proper stopping conditions
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Enable faster inference
    model.eval()
    
    # Define generation function to run in thread
    def generate_in_thread():
        with torch.no_grad(), torch.inference_mode():
            model.generate(**generation_kwargs)
    
    # Start generation in separate thread
    generation_thread = Thread(target=generate_in_thread)
    generation_thread.start()
    
    # Collect streaming output
    full_output = ""
    
    for new_text in streamer:
        if new_text:
            print(new_text, end='', flush=True)
            full_output += new_text
            
            # Count total backticks in the full output
            total_backticks = full_output.count('```')
            
            # Stop if we have 2 or more backticks (code block opened and closed)
            if total_backticks >= 2:
                break
                
            # Also stop if we encounter EOS token
            if tokenizer.eos_token in new_text:
                break
    
    print("\n   ├─ Generation completed")
    print("   └─ Extracting code and plan...")
    print()
    
    # Prepare results
    raw_output = prompt + full_output
    extraction_time = 0
    
    return raw_output, full_output, extraction_time


def _generate_code_non_streaming(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, num_beams: int, do_sample: bool):
    """Generate code without streaming (for beam search or non-sampling)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "num_beams": num_beams,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if num_beams > 1:
        generation_kwargs["early_stopping"] = True
    else:
        # Add optimizations for single-beam generation
        generation_kwargs["use_cache"] = True
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    print(f"   ├─ Using {'beam search' if num_beams > 1 else 'non-streaming'} generation...")
    
    # Enable faster inference
    model.eval()
    with torch.no_grad(), torch.inference_mode():
        outputs = model.generate(**generation_kwargs)
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_output = raw_output[input_length:] if raw_output.startswith(prompt) else raw_output
    print("   ├─ Generation completed")
    print()
    
    return raw_output, full_output


def evaluate_code_detailed(
    code: str, 
    tests: Any, 
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    **kwargs
) -> Dict[str, Any]:
    """Evaluate generated code with detailed feedback."""
    start_time = time.time()
    timeout = kwargs.get('timeout', 30)
    
    try:
        # Parse test cases
        if isinstance(tests, dict):
            inputs = tests.get("inputs", [])
            outputs = tests.get("outputs", [])
            test_cases = list(zip(inputs, outputs))
        else:
            test_cases = [(tc.get("input", ""), tc.get("output", "")) for tc in tests if isinstance(tc, dict)]
        
        passed = 0
        failed = 0
        results = []
        
        for i, (inp, expected_out) in enumerate(test_cases):
            try:
                actual_out = run_code_multi_lang(code, inp, timeout_s=timeout, language=language)
                actual_out_trimmed = actual_out.strip()
                expected_out_trimmed = str(expected_out).strip()
                
                if actual_out_trimmed == expected_out_trimmed:
                    passed += 1
                    results.append({
                        "test": i+1, 
                        "status": "passed", 
                        "input": inp, 
                        "expected": expected_out, 
                        "actual": actual_out
                    })
                else:
                    failed += 1
                    results.append({
                        "test": i+1, 
                        "status": "failed", 
                        "input": inp, 
                        "expected": expected_out, 
                        "actual": actual_out
                    })
                    
            except Exception as e:
                failed += 1
                results.append({
                    "test": i+1, 
                    "status": "error", 
                    "input": inp, 
                    "expected": expected_out, 
                    "error": str(e)
                })
        
        evaluation_time = time.time() - start_time
        
        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "pass_rate": (passed/max(len(test_cases), 1))*100,
            "evaluation_time": evaluation_time,
            "results": results,
            "language": language.value,
            "status": "success" if failed == 0 else "failed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "language": language.value
        }


def test_model_single(
    model, 
    tokenizer: AutoTokenizer,
    problem: Dict[str, Any], 
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON,
    generation_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Test a single model on a problem with structured prompting."""
    if generation_params is None:
        generation_params = {}
    
    tests = problem.get("tests", {})
    
    # Generate code with structured prompting
    generation_result = generate_code(model, tokenizer, problem, language=language, **(generation_params or {}))
    
    if "error" in generation_result:
        return {
            "status": "generation_error",
            "error": generation_result["error"]
        }
    
    generated_code = generation_result["code"]
    
    if not generated_code.strip():
        return {
            "status": "no_code_generated",
            "generation_result": generation_result
        }
    
    # Evaluate code with language-specific execution
    evaluation_result = evaluate_code_detailed(generated_code, tests, language=language)
    
    return {
        "question": problem.get("prompt", problem.get("question", "")),
        "generated_code": generated_code,
        "generated_plan": generation_result.get("plan", ""),
        "generation_result": generation_result,
        "evaluation_result": evaluation_result,
        "language": language.value,
        "status": "completed"
    }