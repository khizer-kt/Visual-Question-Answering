import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class VQAModel(nn.Module):
    def __init__(self, gpt2_model_name="gpt2", num_unfrozen_layers=3):
        super(VQAModel, self).__init__()

        # Load GPT-2 model and tokenizer
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        config = GPT2Config.from_pretrained(gpt2_model_name)

        for param in self.gpt2_model.parameters():
            param.requires_grad = False

        # Unfreeze last few layers
        for name, param in self.gpt2_model.named_parameters():
            if any(layer in name for layer in [f"layer.{i}" for i in range(config.n_layer - num_unfrozen_layers, config.n_layer)]):
                param.requires_grad = True

        # Linear layer to project 1024-dim concatenated features to GPT-2's embedding size
        self.projection = nn.Linear(1024, self.gpt2_model.config.n_embd)

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, combined_features, target_strings=None, max_length=5):
        # Project the combined features to match GPT-2's embedding size
        projected_features = self.projection(combined_features)  # Shape: [batch_size, n_embd]

        # Initialize the GPT-2 model input with the projected features
        model_inputs = projected_features.unsqueeze(1)  # Shape: [batch_size, 1, n_embd]

        generated_tokens = []
        all_logits = []

        for _ in range(max_length):
            outputs = self.gpt2_model(inputs_embeds=model_inputs)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated_tokens.append(next_token)
            all_logits.append(logits)

            # Update model inputs
            next_token_embeds = self.gpt2_model.transformer.wte(next_token)
            model_inputs = torch.cat((model_inputs, next_token_embeds.unsqueeze(1)), dim=1)

            if (next_token == self.gpt2_tokenizer.eos_token_id).all():
                break

        # Stack generated tokens and logits
        generated_tokens = torch.stack(generated_tokens, dim=1)  # Shape: [batch_size, seq_len]
        all_logits = torch.stack(all_logits, dim=1)  # Shape: [batch_size, seq_len, vocab_size]

        # Pad logits to max_length if needed
        if all_logits.size(1) < max_length:
            pad_size = max_length - all_logits.size(1)
            pad_logits = torch.full((all_logits.size(0), pad_size, all_logits.size(2)), float('-inf'), device=all_logits.device)
            all_logits = torch.cat((all_logits, pad_logits), dim=1)

        # Decode generated tokens to get the answer
        decoded_answers = [self.gpt2_tokenizer.decode(tokens, skip_special_tokens=True) for tokens in generated_tokens]

        # Calculate the loss if target strings are provided
        if target_strings is not None:
            # Tokenize the target strings
            target_tokens = self.gpt2_tokenizer(target_strings, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
            target_ids = target_tokens.input_ids.to(combined_features.device)  # Shape: [batch_size, seq_len]

            # Flatten the logits and target ids for loss computation
            loss = self.loss_fn(all_logits.view(-1, all_logits.size(-1)), target_ids.view(-1))
        else:
            loss = None

        return decoded_answers, loss

