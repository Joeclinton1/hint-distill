"""
Custom trainer implementations for hint distillation.
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import Trainer
from torch.utils.data import default_collate

from hint_distill.utils import hint_distill_collate_fn


class HintDistillTrainer(Trainer):
    """Custom trainer implementing hint distillation loss."""
    
    def __init__(self, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T = temperature
        self.alpha = alpha
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=hint_distill_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        stu_out = model(
            input_ids=inputs["input_ids_no_hint"],
            labels=inputs["labels_no_hint"]
        )
        with torch.no_grad():
            tea_out = model(
                input_ids=inputs["input_ids_with_hint"],
                labels=inputs["labels_with_hint"]
            )

        # Get shapes before shifting
        stu_logits = stu_out.logits
        tea_logits = tea_out.logits
        labels = inputs["labels_no_hint"]
        
        # Build shift mask (only compute loss for next token prediction where label exists)
        shift_mask = (labels[:, 1:] != -100)
        
        # Shift logits and labels for causal language modeling
        shifted_stu_logits = stu_logits[:, :-1]
        shifted_tea_logits = tea_logits[:, :-1]
        shifted_labels = labels[:, 1:]
        
        # Check if we have any valid positions after shifting
        if shift_mask.sum() == 0:
            return stu_out.loss
        
        # Compute log probabilities
        s_logprobs = F.log_softmax(shifted_stu_logits / self.T, dim=-1)
        t_probs = F.softmax(shifted_tea_logits / self.T, dim=-1)
        
        # Gather predictions for the actual labels
        batch_size, seq_len, vocab_size = s_logprobs.shape
        
        # Ensure shapes match by using the minimum sequence length
        min_seq_len = min(s_logprobs.shape[1], t_probs.shape[1])
        s_logprobs = s_logprobs[:, :min_seq_len]
        t_probs = t_probs[:, :min_seq_len]
        shift_mask = shift_mask[:, :min_seq_len]
        
        # Reshape for masking
        s_logprobs_reshaped = s_logprobs.reshape(-1, vocab_size)
        t_probs_reshaped = t_probs.reshape(-1, vocab_size)
        shift_mask_flat = shift_mask.reshape(-1)
        
        # Only compute KL divergence on positions where labels exist
        if shift_mask_flat.sum() > 0:
            masked_s_logprobs = s_logprobs_reshaped[shift_mask_flat]
            masked_t_probs = t_probs_reshaped[shift_mask_flat]
            distill = self.kl(masked_s_logprobs, masked_t_probs) * (self.T ** 2)
        else:
            distill = torch.tensor(0.0, device=stu_out.logits.device)
        
        loss = self.alpha * distill + (1 - self.alpha) * stu_out.loss
        return loss


class IntervalValidationTrainer(HintDistillTrainer):
    """Custom trainer that performs validation at intervals within epochs."""
    
    def __init__(self, validation_problems=None, validation_intervals=5, 
                 validation_sample_ratio=0.1, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.validation_problems = validation_problems or []
        self.validation_intervals = validation_intervals
        self.validation_sample_ratio = validation_sample_ratio
        self.tokenizer = tokenizer
        self.steps_per_interval = None
        self.global_step = 0
        
        # Only enable validation if we have validation problems
        self.enable_validation = len(self.validation_problems) > 0
    
    def compute_validation_loss(self, model, val_sample):
        """Compute validation loss on a sample."""
        try:
            model.eval()
            with torch.no_grad():
                # Create a simple dataset-like sample for validation
                val_input_ids = val_sample["input_ids_no_hint"].unsqueeze(0).to(model.device)
                
                # Create labels that won't be ignored - use the last few tokens as target
                val_labels = val_sample["input_ids_no_hint"].clone().unsqueeze(0).to(model.device)
                # Only compute loss for the last portion to simulate actual training
                seq_len = val_labels.shape[1]
                target_len = min(10, seq_len // 4)  # Use last 25% or up to 10 tokens
                if target_len > 0:
                    # Set labels to -100 for all except the target portion
                    val_labels[:, :-target_len] = -100
                else:
                    # If sequence is too short, use the very last token
                    val_labels[:, :-1] = -100
                
                outputs = model(input_ids=val_input_ids, labels=val_labels)
                loss = outputs.loss.item()
                # Only return valid losses (not nan or inf)
                if not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))):
                    return loss
                else:
                    return float('inf')
        except Exception as e:
            print(f"Validation error: {e}")
            return float('inf')
    
    def sample_validation_data(self):
        """Sample validation data for interval validation."""
        if not self.enable_validation or not self.validation_problems:
            return []
        
        # Calculate how many samples to take
        n_samples = max(1, int(len(self.validation_problems) * self.validation_sample_ratio))
        sampled_problems = random.sample(self.validation_problems, min(n_samples, len(self.validation_problems)))
        
        val_samples = []
        for problem in sampled_problems:
            try:
                # Create a simple validation sample similar to training data
                hint = "# Hint: Validate solution"
                no_hint_prompt = problem["prompt"]
                hint_prompt = no_hint_prompt + f"\n{hint}"
                
                # Simple tokenization for validation
                no_hint_ids = self.tokenizer(no_hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                hint_ids = self.tokenizer(hint_prompt, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
                
                # Create dummy labels (simulate training labels)
                labels_no_hint = torch.full_like(no_hint_ids, -100)
                labels_with_hint = torch.full_like(hint_ids, -100)
                
                val_sample = {
                    "input_ids_no_hint": no_hint_ids,
                    "input_ids_with_hint": hint_ids,
                    "labels_no_hint": labels_no_hint,
                    "labels_with_hint": labels_with_hint,
                }
                val_samples.append(val_sample)
            except Exception as e:
                print(f"Error creating validation sample: {e}")
                continue
        
        return val_samples
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to add interval validation."""
        # Perform normal training step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Update global step counter
        self.global_step += 1
        
        # Check if we should perform validation
        if self.enable_validation and self.steps_per_interval is not None:
            if self.global_step % self.steps_per_interval == 0:
                self.perform_interval_validation(model)
        
        return loss
    
    def perform_interval_validation(self, model):
        """Perform validation on a sample of validation data."""
        print(f"Performing interval validation at step {self.global_step}")
        
        val_samples = self.sample_validation_data()
        if not val_samples:
            return
        
        val_losses = []
        for val_sample in val_samples:
            val_loss = self.compute_validation_loss(model, val_sample)
            if val_loss != float('inf'):
                val_losses.append(val_loss)
        
        if val_losses:
            avg_val_loss = np.mean(val_losses)
            print(f"Interval validation loss: {avg_val_loss:.4f}")
            
            # Log to wandb if available
            if hasattr(self, 'args') and hasattr(self.args, 'report_to'):
                if 'wandb' in self.args.report_to:
                    try:
                        import wandb
                        wandb.log({
                            "validation_loss": avg_val_loss,
                            "global_step": self.global_step,
                            "validation_samples": len(val_losses)
                        })
                    except ImportError:
                        pass
    
    def train(self, *args, **kwargs):
        """Override train to set up intervals."""
        if self.enable_validation:
            # Calculate steps per interval based on training dataset size and intervals per epoch
            train_dataset_size = len(self.train_dataset)
            batch_size = self.args.train_batch_size
            steps_per_epoch = (train_dataset_size + batch_size - 1) // batch_size
            
            if self.validation_intervals > 0:
                self.steps_per_interval = max(1, steps_per_epoch // self.validation_intervals)
                print(f"Will perform validation every {self.steps_per_interval} steps ({self.validation_intervals} times per epoch)")
            else:
                self.steps_per_interval = None
                print("Interval validation disabled (validation_intervals = 0)")
        else:
            print("No validation data available, skipping interval validation")
        
        # Call parent train method
        return super().train(*args, **kwargs)