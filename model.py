import torch
import torch.nn as nn

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers):
        super().__init__()

        # Create embedding table
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # Create output linear layer
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, token_ids):
        # embed tokens, embedding dim is a hyperparameter
        hidden = self.embedding(token_ids) # input: [seq_len], output: [seq_len, embedding_dim]
        # logits = hidden_state * W + b
        logits = self.linear(hidden) # input: [seq_len, embedding_dim], output: tensor[seq_len, vocab_size]

        return logits
