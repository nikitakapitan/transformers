"""
The goal of Encoder class is to implement the stack of N layers (EncoderLayer)
test
"""
import torch.nn as nn
from copy import deepcopy
from transformers.LayerNorm import LayerNorm

class Encoder(nn.Module):

    def __init__(self, layer, N):
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)