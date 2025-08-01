#!/usr/bin/env python
"""
hint_distill_demo.py
--------------------
Minimal end‑to‑end demonstration of hint‑distillation fine‑tuning on a single APPS
coding problem using Qwen3‑8B + LoRA (PEFT) with Weights & Biases logging.
"""

import argparse, json
import os
import wandb
from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Disable AWQ loading that causes import errors
os.environ["PEFT_DISABLE_AWQ"] = "1"
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from hint_distill import (
    HintDataset,
    HintDistillTrainer,
    IntervalValidationTrainer,
    FlexibleHintDataset,
    pass_at_k
)
from hint_distill.utils import clear_hint_cache


def create_hint_log_file():
    """Create a unique hint log file for this run."""
    # Ensure hint_logs directory exists
    hint_logs_dir = "hint_logs"
    os.makedirs(hint_logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hint_log_file = os.path.join(hint_logs_dir, f"hint_log_{timestamp}.jsonl")
    return hint_log_file

def log_hint_to_file(hint_log_file, problem_metadata, hint, context, target, hint_method):
    """Log hint information to file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "problem_metadata": problem_metadata,
        "hint": hint,
        "context": context,
        "target": target,
        "hint_method": hint_method
    }
    
    with open(hint_log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def load_apps_examples(start_idx: int, subset_size: int = None):
    """Load multiple examples from APPS dataset."""
    ds = load_dataset("codeparrot/apps", split="test", difficulties=["interview"], trust_remote_code=True)
    
    if subset_size is None:
        # Use full dataset
        end_idx = len(ds)
    else:
        end_idx = min(start_idx + subset_size, len(ds))
    
    problems_data = []
    for i in range(start_idx, end_idx):
        try:
            ex = ds[i]
            tests = json.loads(ex["input_output"])
            solutions = json.loads(ex["solutions"])
            
            problem_data = {
                "prompt": ex["question"],
                "solution": solutions[0] if solutions else "",
                "tests": tests,
                "metadata": {
                    "source": "apps",
                    "index": i,
                    "problem_id": ex.get("problem_id", f"unknown_{i}")
                }
            }
            problems_data.append(problem_data)
        except Exception as e:
            print(f"⚠️  Failed to load example {i}: {e}")
            continue
    
    print(f"✅ Loaded {len(problems_data)} examples from APPS dataset")
    return problems_data


# ---------- Main ---------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="apps", choices=["apps", "json"])
    parser.add_argument("--json_path", help="Custom problem json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--project", default="hint-distill-demo")
    parser.add_argument("--sample_pct", type=float, default=0.0)
    parser.add_argument("--subset_size", type=int, default=None,
                       help="Number of training examples to use (None = full dataset, 1 = old behavior)")
    parser.add_argument("--disable_evals", action="store_true", default=True,
                       help="Disable evaluations to focus on training loss monitoring")
    parser.add_argument("--validation_intervals", type=int, default=5,
                       help="Number of validation intervals per epoch")
    parser.add_argument("--validation_sample_ratio", type=float, default=0.1,
                       help="Fraction of validation set to sample at each interval")
    
    # New hint method parameters
    parser.add_argument("--hint_method", default="self_reflection", choices=["self_reflection", "dataset_solution"],
                       help="Method for generating hints: self_reflection (model attempts + compares to solution) or dataset_solution (direct from solution)")
    
    # Hint cache parameters
    parser.add_argument("--force_regeneration", action="store_true", default=False,
                       help="Force regeneration of all hints, bypassing cache")
    parser.add_argument("--clear_cache", action="store_true", default=False,
                       help="Clear the hint cache before starting")
    
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))

    # Clear cache if requested
    if args.clear_cache:
        print("🗑️  Clearing hint cache...")
        clear_hint_cache()

    # Create hint log file for this run
    hint_log_file = create_hint_log_file()
    print(f"Logging hints to: {hint_log_file}")
    
    # Print cache status
    print(f"🎯 Hint caching enabled (force_regeneration={args.force_regeneration})")

    # Load dataset
    if args.dataset == "apps":
        problems_data = load_apps_examples(args.index, args.subset_size)
        # Use first problem for evaluation reference
        prompt = problems_data[0]["prompt"]
        tests = problems_data[0]["tests"]
    else:
        data = json.load(open(args.json_path))
        problems_data = [{
            "prompt": data["prompt"],
            "solution": data["solution"],
            "tests": data["tests"],
            "metadata": {"source": "json", "path": args.json_path}
        }]
        prompt = data["prompt"]
        tests = data["tests"]

    # Create train/validation split if we have enough data
    train_problems = problems_data
    val_problems = []
    
    if len(problems_data) > 10:
        print(f"Creating train/validation split from {len(problems_data)} problems")
        train_problems, val_problems = train_test_split(
            problems_data, 
            test_size=0.2,  # 20% validation split
            random_state=42
        )
        print(f"Train set: {len(train_problems)} problems, Validation set: {len(val_problems)} problems")
    else:
        print(f"Using all {len(problems_data)} problems for training (validation disabled)")

    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    # Try without quantization first to see if that's the issue
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print("DEBUG: Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    base_model.eval()

    # base_model = prepare_model_for_kbit_training(base_model)  # Skip since no quantization
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Create dataset with selected hint method
    if args.hint_method in ["self_reflection", "dataset_solution"]:
        print(f"Creating FlexibleHintDataset with {len(train_problems)} problems using {args.hint_method} method")
        train_ds = FlexibleHintDataset(
            tok, 
            train_problems, 
            model,  # Pass the model for self-reflection method
            hint_method=args.hint_method,
            hint_log_file=hint_log_file,
            force_regeneration=args.force_regeneration
        )
        
        # Fallback to old dataset if new one fails or for compatibility
        if len(train_ds) == 0:
            print("⚠️  FlexibleHintDataset failed, falling back to HintDataset")
            from hint_distill import ProgrammingLanguage
            # Use first problem for fallback
            fallback_code = train_problems[0]["solution"]
            train_ds = HintDataset(tok, fallback_code, n_lines=3, language=ProgrammingLanguage.PYTHON)
            print(f"DEBUG: Fallback HintDataset created with {len(train_ds)} samples")
    else:
        # For backward compatibility, use original HintDataset
        from hint_distill import ProgrammingLanguage
        fallback_code = train_problems[0]["solution"]
        train_ds = HintDataset(tok, fallback_code, n_lines=3, language=ProgrammingLanguage.PYTHON)
        print(f"Using legacy HintDataset with {len(train_ds)} samples")

    train_args = TrainingArguments(
        output_dir="./hint_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=True,  # Use bf16 instead of fp16 for better compatibility
        logging_steps=1,  # Log every step for detailed loss monitoring
        report_to=["wandb"],
        save_strategy="no",  # Disable saving checkpoints to speed up training
    )

    # Try with low_cpu_mem_usage=False to avoid AWQ issues
    model = get_peft_model(base_model, lora_cfg)
    # Run training
    total_epochs = args.epochs
    
    print(f"Training for {total_epochs} total epochs")
    print(f"Using dataset with {len(train_ds)} samples")
    
    # Create the trainer
    if len(val_problems) > 0:
        print(f"Using IntervalValidationTrainer with {len(val_problems)} validation problems")
        trainer = IntervalValidationTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            validation_problems=val_problems,
            validation_intervals=args.validation_intervals,
            validation_sample_ratio=args.validation_sample_ratio,
            tokenizer=tok,
            temperature=2.0,
            alpha=0.5,
        )
    else:
        print("Using standard HintDistillTrainer (no validation data)")
        trainer = HintDistillTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            temperature=2.0,
            alpha=0.5,
        )
    
    # Run training
    print("Starting training...")
    trainer.train()
    print("DEBUG: Training completed")
    
    # Only run evaluation if not disabled
    if not args.disable_evals:
        print("Running evaluation after training...")
        model.eval()
        finetuned_acc = pass_at_k(model, tok, {"prompt": prompt, "tests": tests}, k=args.k)
        wandb.log({"pass@k_finetuned": finetuned_acc, "epoch": total_epochs})
        print(f"Finetuned model pass@{args.k}: {finetuned_acc}")

        # Evaluate base model for comparison
        base_acc = pass_at_k(base_model, tok, {"prompt": prompt, "tests": tests}, k=args.k)
        wandb.log({"pass@k_base": base_acc})
        print(f"Base model pass@{args.k}: {base_acc}")
    else:
        print("Evaluations disabled - focusing on training loss monitoring")

    wandb.finish()


if __name__ == "__main__":
    main()