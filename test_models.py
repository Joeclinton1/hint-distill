#!/usr/bin/env python
"""
test_models.py
---------------
Simplified testing interface for base and trained LLM checkpoints using the hint_distill module.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Import our hint_distill components
from hint_distill import (
    ModelLoader, 
    ProblemLoader,
    generate_code,
    evaluate_code_detailed,
    test_model_single,
    ProgrammingLanguage
)


class ModelTester:
    """Simplified model testing interface using hint_distill components."""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        
    def load_models(self, model_name: str, checkpoint_path: str = None) -> bool:
        """Load models using ModelLoader."""
        print(f"🔄 Loading models...")
        
        if not self.model_loader.load_base_model(model_name):
            return False
        
        if checkpoint_path:
            if not self.model_loader.load_trained_model(checkpoint_path, model_name):
                print(f"⚠️  Warning: Failed to load trained model from {checkpoint_path}")
        
        return True
    
    def load_problem(self, apps_index: int = None, json_path: str = None, difficulty: str = "competition"):
        """Load problem using ProblemLoader."""
        if apps_index is not None:
            return ProblemLoader.load_from_apps(apps_index, difficulty)
        else:
            return ProblemLoader.load_from_json(json_path)
    
    def test_model_with_ui(self, model_name: str, problem: Dict[str, Any], language: ProgrammingLanguage, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single model with UI feedback."""
        base_model, trained_model, tokenizer = self.model_loader.get_models()
        model = base_model if model_name == "Base Model" else trained_model
        
        if model is None:
            print(f"❌ {model_name} not loaded")
            return {"model": model_name, "status": "not_loaded"}
        
        print(f"🎯 **Testing {model_name} ({language.value.upper()})**")
        print("=" * 80)
        
        question = problem.get("prompt", problem.get("question", ""))
        print(f"📋 **Problem Description:**")
        print(question[:500] + "..." if len(question) > 500 else question)
        print()
        
        # Enable streaming by default
        if generation_params is None:
            generation_params = {}
        generation_params.setdefault("stream", True)
        generation_params.setdefault("num_beams", 1)  # Streaming works best with single beam
        
        # Generate code with structured prompting and streaming
        print("🧠 Generating structured response (plan + code)...")
        print("   │")
        print("   │ 🚀 Starting model inference...")
        print()
        
        if tokenizer is None:
            print("❌ Tokenizer not available")
            return {"model": model_name, "status": "tokenizer_error"}
        
        generation_result = generate_code(model, tokenizer, problem, language=language, **generation_params)
        
        if "error" in generation_result:
            print(f"❌ Code generation failed: {generation_result['error']}")
            return {"model": model_name, "status": "generation_error", "error": generation_result["error"]}
        
        generated_code = generation_result["code"]
        generated_plan = generation_result.get("plan", "")
        
        if not generated_code.strip():
            print("❌ No code generated")
            return {"model": model_name, "status": "no_code_generated"}
        
        # Display the generated plan if available
        if generated_plan.strip():
            print("📋 **Generated Plan:**")
            print("---")
            print(generated_plan)
            print("---")
            print()
        
        print("📝 **Generated Code:**")
        print(f"```{language.value}")
        lines = generated_code.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line if line.strip() else ''}")
        print("```")
        print()
        
        print("📊 **Generation Metrics:**")
        print(f"   ├─ Language: {language.value}")
        print(f"   ├─ Generation time: {generation_result['generation_time']:.2f}s")
        print(f"   ├─ Tokens generated: {generation_result['tokens']}")
        print(f"   ├─ Code lines: {generation_result['lines']}")
        print(f"   ├─ Streaming: {'✅ Enabled' if generation_params.get('stream') else '❌ Disabled'}")
        print()
        
        # Evaluate code with language-specific execution
        print("🧪 **Evaluating generated code...**")
        evaluation_result = evaluate_code_detailed(generated_code, problem.get("tests", {}), language=language)
        
        print("📈 **Evaluation Summary:**")
        print(f"   ├─ Total tests: {evaluation_result.get('total', 0)}")
        print(f"   ├─ Passed: {evaluation_result.get('passed', 0)}")
        print(f"   ├─ Failed: {evaluation_result.get('failed', 0)}")
        print(f"   ├─ Pass rate: {evaluation_result.get('pass_rate', 0):.1f}%")
        print(f"   ├─ Evaluation time: {evaluation_result.get('evaluation_time', 0):.2f}s")
        print(f"   └─ Overall: {'✅ SUCCESS' if evaluation_result.get('status') == 'success' else '❌ FAILED'}")
        print()
        
        return {
            "model": model_name,
            "question": question,
            "generated_code": generated_code,
            "generated_plan": generated_plan,
            "generation_result": generation_result,
            "evaluation_result": evaluation_result,
            "language": language.value,
            "status": "completed"
        }
    
    def compare_models_ui(self, problem: Dict[str, Any], language: ProgrammingLanguage = ProgrammingLanguage.PYTHON, generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare models with UI feedback."""
        if generation_params is None:
            generation_params = {}
            
        print(f"🔄 **Comparing Base vs Trained Models ({language.value.upper()})**")
        print("=" * 80)
        
        base_model, trained_model, _ = self.model_loader.get_models()
        results = {}
        
        # Test base model
        if base_model is not None:
            results["base"] = self.test_model_with_ui("Base Model", problem, language, generation_params or {})
        
        # Test trained model
        if trained_model is not None:
            results["trained"] = self.test_model_with_ui("Trained Model", problem, language, generation_params or {})
        
        # Summary comparison
        print("📊 **Comparison Summary**")
        print("=" * 80)
        
        for model_type, result in results.items():
            if result["status"] == "completed":
                eval_result = result["evaluation_result"]
                gen_result = result["generation_result"]
                
                print(f"**{model_type.upper()} MODEL:**")
                print(f"  ├─ Language: {result.get('language', 'unknown')}")
                print(f"  ├─ Code Generation: ✅")
                print(f"  ├─ Generated lines: {gen_result['lines']}")
                print(f"  ├─ Generation time: {gen_result['generation_time']:.2f}s")
                print(f"  ├─ Test cases passed: {eval_result.get('passed', 0)}/{eval_result.get('total', 0)}")
                print(f"  ├─ Pass rate: {eval_result.get('pass_rate', 0):.1f}%")
                print(f"  ├─ Evaluation time: {eval_result.get('evaluation_time', 0):.2f}s")
                print(f"  └─ Status: {eval_result.get('status', 'unknown')}")
                print()
        
        return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save testing results to file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Test base and trained LLM models on programming problems")
    
    # Model options
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Base model name")
    parser.add_argument("--checkpoint", help="Path to trained model checkpoint")
    
    # Problem options
    problem_group = parser.add_mutually_exclusive_group(required=True)
    problem_group.add_argument("--apps_index", type=int, help="APPS dataset problem index")
    problem_group.add_argument("--json_path", help="Path to JSON problem file")
    
    parser.add_argument("--difficulty", default="competition", 
                       choices=["introductory", "interview", "competition"], 
                       help="APPS problem difficulty")
    
    # Language options
    parser.add_argument("--language", default="python", 
                       choices=["python", "cpp", "java", "javascript"], 
                       help="Programming language for code generation and evaluation")
    
    # Generation options
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling")
    parser.add_argument("--no_stream", action="store_false", dest="stream", default=True, help="Disable streaming generation")
    
    # Output options
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--save_code", help="Save generated code to directory")
    
    args = parser.parse_args()
    
    # Map language string to enum
    try:
        language = ProgrammingLanguage(args.language.lower())
    except ValueError:
        print(f"❌ Unsupported language: {args.language}. Using Python as default.")
        language = ProgrammingLanguage.PYTHON
    
    # Initialize tester
    tester = ModelTester()
    
    # Load models
    if not tester.load_models(args.model_name, args.checkpoint):
        return 1
    
    # Load problem
    if args.apps_index is not None:
        problem = tester.load_problem(args.apps_index, difficulty=args.difficulty)
        if problem is None:
            problem = tester.load_problem(args.apps_index, difficulty=args.difficulty, json_path="")
    else:
        problem = tester.load_problem(json_path=args.json_path)
    
    if not problem:
        return 1
    
    # Generation parameters
    generation_params = {
        "max_new_tokens": args.max_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "stream": args.stream
    }
    
    # Run comparison
    results = tester.compare_models_ui(problem, language, generation_params)
    
    # Save results
    if args.output:
        save_results(results, args.output)
    
    # Save generated codes
    if args.save_code:
        save_dir = Path(args.save_code)
        save_dir.mkdir(exist_ok=True)
        
        for model_name, result in results.items():
            if result["status"] == "completed":
                language_ext = result.get("language", "python")
                code_path = save_dir / f"{model_name}_solution.{language_ext}"
                with open(code_path, 'w') as f:
                    f.write(result["generated_code"])
                print(f"💾 Saved {model_name} code to: {code_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())