"""
ResidualConnection implements residual connection to any sublayer!

it also applies LayerNorm after 'sublayer'
it also applies DropOut after LayerNorm
"""

import torch.nn as nn
from typing import Callable # python explicit typin
from transformers.LayerNorm import LayerNorm

class ResidualConnection(nn.Module):

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, sublayer : Callable):
        return x + self.dropout(self.norm(sublayer(x)))