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
from datasets import load_dataset

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
    FlexibleHintDataset,
    pass_at_k
)


def load_apps_examples(start_idx: int, subset_size: int = None):
    """Load multiple examples from APPS dataset."""
    ds = load_dataset("codeparrot/apps", split="test[:100]", difficulties=["interview"], trust_remote_code=True)
    
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
    
    # New hint method parameters
    parser.add_argument("--hint_method", default="self_reflection", choices=["self_reflection", "dataset_solution"],
                       help="Method for generating hints: self_reflection (model attempts + compares to solution) or dataset_solution (direct from solution)")
    
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))

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
        print(f"Creating FlexibleHintDataset with {len(problems_data)} problems using {args.hint_method} method")
        train_ds = FlexibleHintDataset(
            tok, 
            problems_data, 
            model,  # Pass the model for self-reflection method
            hint_method=args.hint_method
        )
        
        # Fallback to old dataset if new one fails or for compatibility
        if len(train_ds) == 0:
            print("⚠️  FlexibleHintDataset failed, falling back to HintDataset")
            from hint_distill import ProgrammingLanguage
            # Use first problem for fallback
            fallback_code = problems_data[0]["solution"]
            train_ds = HintDataset(tok, fallback_code, n_lines=3, language=ProgrammingLanguage.PYTHON)
            print(f"DEBUG: Fallback HintDataset created with {len(train_ds)} samples")
    else:
        # For backward compatibility, use original HintDataset
        from hint_distill import ProgrammingLanguage
        fallback_code = problems_data[0]["solution"]
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