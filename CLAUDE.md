# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is Hint Distill - a self-supervised hint distillation system for improving LLM performance on competitive programming tasks. The project finetunes Qwen3-8B using LoRA to predict helpful hints for next code blocks.

## Architecture
- **Single-file demo** (`hint_distill_demo.py`) containing end-to-end training pipeline
- **Distillation mechanism**: Teacher (with hints) → Student (without hints) via KL divergence
- **Hint generation**: 3-line code chunks with auto-generated hints based on first non-empty line
- **Evaluation**: pass@k metric using code execution against test cases

## Core Components
1. **HintDataset**: Creates training samples from code chunks with/without hints
2. **HintDistillTrainer**: Custom trainer implementing hint distillation loss (α×distill + (1-α)×CE)
3. **Code execution**: Safe subprocess execution for testing generated code
4. **APPS integration**: Uses `load_dataset("codeparrot/apps")` for training data

## Commands
- Install: `pip install -r requirements.txt`
- Run demo: `python hint_distill_demo.py --dataset apps --index 0 --k 5 --epochs 1`
- Custom JSON: `python hint_distill_demo.py --dataset json --json_path path/to/problem.json`
- Logging: `wandb login` (required for Weights & Biases)

## Key Parameters
- `--k`: Number of attempts for pass@k evaluation (default: 5)
- `--epochs`: Training epochs for LoRA finetuning (default: 1)
- `--sample_pct`: Percentage of training data to use (default: 0.0 - use all)

## File Structure
- `hint_distill_demo.py`: Complete training pipeline (~200 lines)
- `requirements.txt`: PyTorch, transformers, peft, wandb dependencies
- `README.md`: Basic usage instructions