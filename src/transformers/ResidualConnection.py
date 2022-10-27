"""
SublayerConnection implements residual connection to any sublayer!

it also normalize the output of sublayer.
"""

import torch.nn as nn
from LayerNorm import LayerNorm
from typing import Callable # python explicit typin

class ResidualConnection(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.norm = LayerNorm(size)
        
        
    def forward(self, x, sublayer : Callable):
        return x + self.norm(sublayer(x))