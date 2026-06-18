import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import round_nestedList


class BERTWrapper(nn.Module):
    """
    Wrapper for BERT models for multiple choice classification.
    Scores each choice by encoding [CLS] input [SEP] choice [SEP].
    """

    def __init__(self, transformer, tokenizer):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(transformer.config.hidden_size, 1)

    def _score_choices(self, input_ids, input_mask, choices_ids, choices_mask):
        """
        Score all (input, choice) pairs.
        
        Args:
            input_ids: [batch_size, max_input_len]
            input_mask: [batch_size, max_input_len]  
            choices_ids: [batch_size * num_choices, max_choice_len]
            choices_mask: [batch_size * num_choices, max_choice_len]
            
        Returns:
            scores: [batch_size, num_choices]
        """
        batch_size = input_ids.shape[0]
        num_choices = choices_ids.shape[0] // batch_size
        
        # Repeat inputs for each choice
        input_ids_repeated = torch.repeat_interleave(input_ids, num_choices, dim=0)
        input_mask_repeated = torch.repeat_interleave(input_mask, num_choices, dim=0)
        
        # Get actual sequence lengths
        input_lengths = input_mask_repeated.sum(dim=1)
        choice_lengths = choices_mask.sum(dim=1)
        
        # Concatenate: [CLS] input [SEP] choice [SEP]
        max_total_len = input_ids.shape[1] + choices_ids.shape[1]
        combined_ids = torch.full(
            (input_ids_repeated.shape[0], max_total_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        combined_mask = torch.zeros(
            (input_ids_repeated.shape[0], max_total_len),
            dtype=torch.long,
            device=input_ids.device
        )
        token_type_ids = torch.zeros_like(combined_ids)
        
        for i in range(input_ids_repeated.shape[0]):
            inp_len = input_lengths[i].item()
            choice_len = choice_lengths[i].item()
            
            # Copy input (already has [CLS] and [SEP])
            combined_ids[i, :inp_len] = input_ids_repeated[i, :inp_len]
            combined_mask[i, :inp_len] = 1
            
            # Skip [CLS] from choice if present
            choice_start = 1 if choices_ids[i, 0] == self.tokenizer.cls_token_id else 0
            choice_actual_len = choice_len - choice_start
            
            # Append choice tokens
            combined_ids[i, inp_len:inp_len + choice_actual_len] = choices_ids[i, choice_start:choice_len]
            combined_mask[i, inp_len:inp_len + choice_actual_len] = 1
            token_type_ids[i, inp_len:inp_len + choice_actual_len] = 1
        
        # Pass through BERT
        outputs = self.transformer(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            token_type_ids=token_type_ids,
        )
        
        # Score using [CLS] representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        scores = self.classifier(cls_output).squeeze(-1)
        scores = scores.view(batch_size, num_choices)
        
        return scores

    def forward(self, batch):
        """Training forward pass."""
        scores = self._score_choices(
            batch["input_ids"],
            batch["input_mask"],
            batch["all_choices_ids"],
            batch["all_choices_mask"],
        )
        
        loss = F.cross_entropy(scores, batch["lbl"])
        return loss, {"loss": loss.detach().cpu().item()}

    def predict_mulChoice(self, batch, length_normalization=None):
        """Prediction for multiple choice."""
        scores = self._score_choices(
            batch["input_ids"],
            batch["input_mask"],
            batch["all_choices_ids"],
            batch["all_choices_mask"],
        )
        
        batch_size = scores.shape[0]
        num_choices = scores.shape[1]
        
        # Apply softmax to get probabilities
        probs = F.softmax(scores, dim=1)
        
        # Get predicted choice
        predicted_choice = torch.argmax(scores, dim=1)
        
        # Create empty placeholders with correct structure for compatibility
        logprobs_placeholder = [[[] for _ in range(num_choices)] for _ in range(batch_size)]
        lengths_placeholder = [[0 for _ in range(num_choices)] for _ in range(batch_size)]
        
        return (
            predicted_choice.cpu().numpy().tolist(),
            round_nestedList(probs.cpu().numpy().tolist(), 5),
            logprobs_placeholder,
            lengths_placeholder,
        )

    def generate(self, batch, max_generationLength):
        """BERT cannot generate text."""
        raise NotImplementedError("BERT cannot generate text. Use T5 for generation.")
