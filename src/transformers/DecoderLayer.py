"""
DecoderLayer implements 3 parts:
1. Masked multi-head self-attention + normalization + residual connection
2. Multi-head self-attention + normalization + residual connection
3. position-wise fully connected feed-forward + normalization + residual connection
"""
import torch as nn
import numpy as np
from transformers.helper import clones
from transformers.LayerNorm import LayerNorm
from transformers.ResidualConnection import ResidualConnection

class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_fwd, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_fwd = feed_fwd
        self.resconnect = clones(ResidualConnection(size), 2)
        
        
        self.norm = LayerNorm(size)
        self.droput = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        self_attn_func = lambda x : self.self_attn(x, x, x, mask)
        
        x = self.resconnect[0](x, sublayer=self_attn_func)
        x = self.resconnect[1](x, sublayer=self.feed_fwd)
        return x
