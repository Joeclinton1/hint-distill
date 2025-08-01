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


def load_apps_example(idx: int):
    """Load a single example from APPS dataset."""
    ds = load_dataset("codeparrot/apps", split="test[:100]", difficulties=["competition"], trust_remote_code=True)
    ex = ds[idx]
    tests = json.loads(ex["input_output"])
    solutions = json.loads(ex["solutions"])
    return ex["question"], solutions[0], tests


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
    
    # New hint method parameters
    parser.add_argument("--hint_method", default="self_reflection", choices=["self_reflection", "dataset_solution"],
                       help="Method for generating hints: self_reflection (model attempts + compares to solution) or dataset_solution (direct from solution)")
    
    args = parser.parse_args()

    print("DEBUG: Initializing wandb...")
    wandb.init(project=args.project, config=vars(args))
    print("DEBUG: wandb initialized")

    if args.dataset == "apps":
        prompt, gt_code, tests = load_apps_example(args.index)
    else:
        data = json.load(open(args.json_path))
        prompt, gt_code, tests = data["prompt"], data["solution"], data["tests"]

    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_name)
    # Try without quantization first to see if that's the issue
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print("DEBUG: Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    print("DEBUG: Model loaded, calling eval()...")
    base_model.eval()

    print("DEBUG: Model eval completed, preparing for training...")
    # base_model = prepare_model_for_kbit_training(base_model)  # Skip since no quantization
    print("DEBUG: Model ready for training")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    # Create problem data for the new dataset
    if args.dataset == "apps":
        problem_data = {
            "prompt": prompt,
            "solution": gt_code,
            "tests": tests,
            "metadata": {"source": "apps", "index": args.index}
        }
    else:
        problem_data = {
            "prompt": prompt,
            "solution": gt_code,
            "tests": tests,
            "metadata": {"source": "json", "path": args.json_path}
        }
    
    # Create dataset with selected hint method
    print("DEBUG: Creating FlexibleHintDataset...")
    train_ds = FlexibleHintDataset(
        tok, 
        [problem_data], 
        model,  # Pass the model for self-reflection method
        hint_method=args.hint_method
    )
    print(f"DEBUG: FlexibleHintDataset created with {len(train_ds)} samples")
    
    # Fallback to old dataset if new one fails or for compatibility
    if len(train_ds) == 0:
        print("⚠️  FlexibleHintDataset failed, falling back to HintDataset")
        from hint_distill import ProgrammingLanguage
        train_ds = HintDataset(tok, gt_code, n_lines=3, language=ProgrammingLanguage.PYTHON)
        print(f"DEBUG: Fallback HintDataset created with {len(train_ds)} samples")

    train_args = TrainingArguments(
        output_dir="./hint_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=True,  # Use bf16 instead of fp16 for better compatibility
        logging_steps=1,
        report_to=["wandb"],
    )

    print("DEBUG: Creating PEFT model...")
    # Try with low_cpu_mem_usage=False to avoid AWQ issues
    model = get_peft_model(base_model, lora_cfg)
    print("DEBUG: PEFT model created")

    print("DEBUG: Creating HintDistillTrainer...")
    trainer = HintDistillTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        temperature=2.0,
        alpha=0.5,
    )
    print("DEBUG: Trainer created, starting training...")
    trainer.train()
    print("DEBUG: Training completed")

    model.eval()
    finetuned_acc = pass_at_k(model, tok, {"prompt": prompt, "tests": tests}, k=args.k)
    wandb.log({"pass@k_finetuned": finetuned_acc})
    print(f"Finetuned model pass@{args.k}: {finetuned_acc}")

    # Evaluate base model after training for comparison
    base_acc = pass_at_k(base_model, tok, {"prompt": prompt, "tests": tests}, k=args.k)
    wandb.log({"pass@k_base": base_acc})
    print(f"Base model pass@{args.k}: {base_acc}")

    wandb.finish()


if __name__ == "__main__":
    main()