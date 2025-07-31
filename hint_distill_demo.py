#!/usr/bin/env python
"""
hint_distill_demo.py
--------------------
Minimal end‑to‑end demonstration of hint‑distillation fine‑tuning on a single APPS
coding problem using Qwen3‑8B + LoRA (PEFT) with Weights & Biases logging.
"""

import argparse, os, json, random, subprocess, tempfile, textwrap, pathlib, signal, time, sys
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import wandb


# ---------- Utility: run user code safely ---------- #
def run_python(code_str: str, inp: str, timeout_s: int = 2) -> str:
    """Executes `code_str` in a subprocess, feeds `inp` to stdin, returns stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str)
        tmp_path = tmp.name
    cmd = [sys.executable, tmp_path]
    try:
        proc = subprocess.run(
            cmd,
            input=inp.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        out = proc.stdout.decode().strip()
    except subprocess.TimeoutExpired:
        out = "__TIMEOUT__"
    finally:
        os.remove(tmp_path)
    return out


# ---------- APPS helpers ---------- #
def load_apps_example(idx: int):
    ds = load_dataset("codeparrot/apps", split="test[:100]")  # lightweight subset
    ex = ds[idx]
    tests = json.loads(ex["input_output"])
    return ex["question"], ex["solutions"][0], tests


def tests_pass(code_str: str, tests, timeout=2):
    for io_pair in tests:
        _in, _out = io_pair["input"], io_pair["output"].strip()
        pred = run_python(code_str, _in, timeout)
        if pred.strip() != _out:
            return False
    return True


# ---------- Hint creation ---------- #
def make_chunks(code: str, n_lines: int = 3):
    lines = code.splitlines()
    for i in range(0, len(lines), n_lines):
        yield lines[:i], lines[i : i + n_lines]


def auto_hint(ctx_lines, tgt_lines):
    stripped = [l.strip() for l in tgt_lines if l.strip()]
    return f"# Hint: {stripped[0][:60]}" if stripped else "# Hint: next steps"


# ---------- Custom dataset ---------- #
class HintDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, code, n_lines=3):
        self.samples = []
        for ctx, tgt in make_chunks(code, n_lines):
            ctx_src = "\n".join(ctx)
            tgt_src = "\n".join(tgt)
            hint = auto_hint(ctx, tgt)

            with_hint = tokenizer(ctx_src + "\n" + hint, return_tensors="pt").input_ids[0]
            no_hint = tokenizer(ctx_src, return_tensors="pt").input_ids[0]
            tgt_ids = tokenizer(tgt_src, return_tensors="pt").input_ids[0]

            labels_with_hint = torch.full_like(with_hint, -100)
            labels_with_hint[-len(tgt_ids):] = tgt_ids

            labels_no_hint = torch.full_like(no_hint, -100)
            labels_no_hint[-len(tgt_ids):] = tgt_ids

            self.samples.append({
                "input_ids_with_hint": with_hint,
                "input_ids_no_hint": no_hint,
                "labels_with_hint": labels_with_hint,
                "labels_no_hint": labels_no_hint,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------- Distillation Trainer ---------- #
class HintDistillTrainer(Trainer):
    def __init__(self, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature
        self.alpha = alpha
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        stu_out = model(
            input_ids=inputs["input_ids_no_hint"],
            labels=inputs["labels_no_hint"]
        )
        with torch.no_grad():
            tea_out = model(
                input_ids=inputs["input_ids_with_hint"],
                labels=inputs["labels_with_hint"]
            )

        mask = inputs["labels_no_hint"] != -100
        s_logprobs = F.log_softmax(stu_out.logits / self.T, dim=-1)
        t_probs = F.softmax(tea_out.logits / self.T, dim=-1)
        distill = self.kl(s_logprobs[mask], t_probs[mask]) * (self.T ** 2)
        loss = self.alpha * distill + (1 - self.alpha) * stu_out.loss
        return loss


# ---------- Evaluation ---------- #
def pass_at_k(model, tokenizer, prompt, tests, k=5, temp=0.2, max_new=256):
    ok = False
    for _ in range(k):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            do_sample=True,
            temperature=temp,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
        )
        code = tokenizer.decode(gen[0], skip_special_tokens=True)
        if code.startswith(prompt):
            code = code[len(prompt):]
        if tests_pass(code, tests):
            ok = True
            break
    return 1.0 if ok else 0.0


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
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))

    if args.dataset == "apps":
        prompt, gt_code, tests = load_apps_example(args.index)
    else:
        data = json.load(open(args.json_path))
        prompt, gt_code, tests = data["prompt"], data["solution"], data["tests"]

    model_name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=True, device_map="auto"
    )
    base_model.eval()

    base_acc = pass_at_k(base_model, tok, prompt, tests, k=args.k)
    wandb.log({"pass@k_base": base_acc})
    print(f"Base model pass@{args.k}: {base_acc}")

    base_model = prepare_model_for_int8_training(base_model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    train_ds = HintDataset(tok, gt_code, n_lines=3)

    train_args = TrainingArguments(
        output_dir="./hint_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        report_to=["wandb"],
    )

    trainer = HintDistillTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        temperature=2.0,
        alpha=0.5,
    )
    trainer.train()

    model.eval()
    finetuned_acc = pass_at_k(model, tok, prompt, tests, k=args.k)
    wandb.log({"pass@k_finetuned": finetuned_acc})
    print(f"Finetuned model pass@{args.k}: {finetuned_acc}")

    wandb.finish()


if __name__ == "__main__":
    main()