"""
Decoder class implements the stack of N layers (DecoderLayer)
"""
from typing import Callable
import torch.nn as nn
from copy import deepcopy
from LayerNorm import LayerNorm

class Decoder(nn.Module):

    def __init__(self, layer : Callable, N):
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)