"""
Model loading utilities for hint distillation.
"""

import os
import time
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel

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


class ModelLoader:
    """Handles loading of base and fine-tuned models."""
    
    def __init__(self):
        self.base_model = None
        self.trained_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_base_model(self, model_name: str = "Qwen/Qwen3-4B", 
                       quantization: bool = True) -> bool:
        """Load base model with optional quantization."""
        start_time = time.time()
        
        try:
            if quantization:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Enable faster inference
            self.base_model.eval()
            # Warm up the model
            try:
                device = next(self.base_model.parameters()).device
                dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
                with torch.no_grad():
                    _ = self.base_model(dummy_input)
            except:
                pass  # Skip warm-up if there are issues
            
            load_time = time.time() - start_time
            print(f"✅ Base model loaded in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load base model: {e}")
            return False
    
    def load_trained_model(self, checkpoint_path: str, 
                          base_model_name: str = "Qwen/Qwen3-4B",
                          quantization: bool = True) -> bool:
        """Load fine-tuned model from LoRA checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        start_time = time.time()
        
        try:
            # Load base model
            if quantization:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load LoRA adapter
            self.trained_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            # Enable faster inference
            self.trained_model.eval()
            # Warm up the model
            try:
                device = next(self.trained_model.parameters()).device
                dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
                with torch.no_grad():
                    _ = self.trained_model(dummy_input)
            except:
                pass  # Skip warm-up if there are issues
            
            load_time = time.time() - start_time
            print(f"✅ Trained model loaded in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            return False
    
    def get_models(self) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[AutoTokenizer]]:
        """Get loaded models and tokenizer."""
        return self.base_model, self.trained_model, self.tokenizer