"""
Custom trainer implementations for hint distillation.
"""

import torch
import torch.nn.functional as F
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