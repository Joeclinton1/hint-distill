"""
Hint Distill - A self-supervised hint distillation system for competitive programming.
"""

from .data import HintDataset, MultiLanguageHintDataset, ProblemLoader, SelfReflectionHintDataset, FlexibleHintDataset
from .trainer import HintDistillTrainer, IntervalValidationTrainer
from .models import ModelLoader
from .evaluation import (
    pass_at_k, tests_pass, generate_code, evaluate_code_detailed, test_model_single
)
from .utils import run_python, run_code_multi_lang, make_chunks, auto_hint, hint_distill_collate_fn
from .prompting import (
    PromptTemplates, CodeExtractor, ProgrammingLanguage, LanguageUtils
)

__version__ = "0.1.0"
__all__ = [
    "HintDataset",
    "MultiLanguageHintDataset",
    "ProblemLoader",
    "SelfReflectionHintDataset",
    "FlexibleHintDataset",
    "HintDistillTrainer",
    "IntervalValidationTrainer",
    "ModelLoader",
    "pass_at_k",
    "tests_pass",
    "generate_code",
    "evaluate_code_detailed", 
    "test_model_single",
    "run_python",
    "run_code_multi_lang",
    "make_chunks",
    "auto_hint",
    "generate_self_reflection_hint",
    "generate_hint",
    "generate_dataset_solution_hint",
    "hint_distill_collate_fn",
    "PromptTemplates",
    "CodeExtractor", 
    "ProgrammingLanguage",
    "LanguageUtils"
]