from __future__ import annotations

import torch
import torch.nn as nn
import torch.functional as F

from ..configs import Config

def scaled_dot_product(Q, K, V, mask=None):
    dk = Q.shape[-1]
    logits = torch.matmul(Q, K.transpose(-2, -1))
    logits = logits / torch.sqrt(dk)
    if mask is not None:
        logits = logits.masked_fill(mask == 0, -1e9)
    weights = F.softmax(logits, dim=-1)
    values = torch.matmul(weights, V)
    return values, weights

class MultiHeadAttention:
    def __init__(self, n_heads, embed_dim, input_dim):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.QW = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.KW = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.VW = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        
    def 
class Transformer(nn.Module):
    def __init__(self, n_heads):
        self.n_heads = n_heads

    