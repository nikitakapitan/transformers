
"""
EncoderLayer implements two parts:
1. multi-head self-attention
2. position-wise fully connected feed-forward
"""

import torch.nn  as nn
from ResidualConnection import ResidualConnection
from typing import Callable
from LayerNorm import LayerNorm

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn : Callable, feed_fwd, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_fwd = feed_fwd
        self.resconnect = ResidualConnection(size)
        self.size = size
        
        self.norm = LayerNorm(size)
        self.droput = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        self_attn_func = lambda x : self.self_attn(x, x, x, mask)
        
        x = self.resconnect(x, sublayer=self_attn_func)
        x = self.resconnect(x, sublayer=self.feed_fwd)
        return x