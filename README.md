# Hint Distill

Self-supervised hint distillation for improving LLMs on code generation.  
This project finetunes Qwen3-4B on competitive programming datasets by generating a "hint" based on the code solutuion, and then using KL-divergence based distillation between the model with access to the hint acting as the teacher for itself.
## Installation

```bash
git clone https://github.com/Joeclinton1/hint-distill.git
cd hint-distill
conda create -n hint-distill python=3.10 -y && conda activate hint-distill
```

**Install PyTorch** (choose your CUDA version):
👉 [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

Then:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the full hint distillation pipeline on one APPS problem:

```bash
python hint_distill_demo.py --dataset apps --index 0 --k 5 --epochs 1
```

Or use a custom JSON file:

```bash
python hint_distill_demo.py --dataset json --json_path path/to/problem.json
```

Each JSON should have:

```json
{
  "prompt": "...",
  "solution": "...",
  "tests": [{"input": "...", "output": "..."}]
}
```

Log in to Weights & Biases with:

```bash
wandb login
```

---

## License

MIT
