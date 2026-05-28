import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers):
        super().__init__()

        # Create embedding table
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        # Create transformer blocks(stack num layers of them)
        self.transformers = torch.nn.ModuleList([TransformerBlock(embedding_dim) for i in range(num_layers)])

        # Create output linear layer
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, token_ids):
        # embed tokens, embedding dim is a hyperparameter
        hidden = self.embedding(token_ids) # input: [seq_len], output: [seq_len, embedding_dim]

        # pass through each transformer block
        for block in self.transformers:
            hidden = block(hidden)
        # logits = hidden_state * W + b
        logits = self.linear(hidden) # input: [seq_len, embedding_dim], output: tensor[seq_len, vocab_size]

        return logits

class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # create w_q, w_k and w_v as linear layers
        # each projects from embedding dim to embedding dim
        self.embedding_dim = embedding_dim
        self.W_q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.W_k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.W_v = torch.nn.Linear(embedding_dim, embedding_dim)
        

    def forward(self, hidden):
        # project hidden to q, k ,v using linear layers
        Q = self.W_q(hidden)
        K = self.W_k(hidden)
        V = self.W_v(hidden)

        # compute scores = q * k^T
        scores = Q @ K.T
        # scale by sqrt(embedding_dim)
        scores = scores / math.sqrt(self.embedding_dim)
        # create mask for causal masking
        seq_len = hidden.shape[0]
        mask = torch.tril(torch.ones(seq_len, seq_len)) # lower triangular matrix: 1 for valid positions and 0 for future positions
        #apply mask
        scores = scores.masked_fill(mask == 0, -1e9) # set to very negative number
        # apply softmax
        attention_probs = torch.softmax(scores, dim=-1)
        # multiply by V
        output = attention_probs @ V
        # return output
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        #create an Attention layer
        self.attention = Attention(embedding_dim)
        #create a Linear layer (first part of FFN) expansion
        self.FFN_1 = torch.nn.Linear(embedding_dim, embedding_dim * 4)
        #create another Linear layer (second part of FFN) contraction
        self.FFN_2 = torch.nn.Linear(embedding_dim * 4, embedding_dim)
        #create LayerNorm layers (we'll use 2 of them)
        self.layer_norm_1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(embedding_dim)
        
    
    def forward(self, hidden):
        # Self-attention + residual connection
        attn_output = self.attention(hidden)
        hidden = hidden + attn_output  #(residual connection)
        hidden = self.layer_norm_1(hidden)
        
        # FFN + residual connection
        ffn_output = self.FFN_1(hidden)
        ffn_output = F.relu(ffn_output)
        ffn_output = self.FFN_2(ffn_output)
        hidden = hidden + ffn_output  
        hidden = self.layer_norm_2(hidden)
        
        return hidden