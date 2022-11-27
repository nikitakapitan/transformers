
"""
EncoderLayer implements two parts:
1. multi-head self-attention
2. position-wise fully connected feed-forward
"""

import torch.nn  as nn
from typing import Callable

from transformers.helper import clones
from transformers.ResidualConnection import ResidualConnection
from transformers.LayerNorm import LayerNorm

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn : Callable, feed_fwd, dropout):
        super().__init__()
        self.self_attn = self_attn # MultiHeadedAttention
        self.feed_fwd = feed_fwd
        self.resconnect = clones(ResidualConnection(size, dropout), 2)
        self.size = size
        # note : LayerNorm and DropOut are applied within 'resconnect'

        
    def forward(self, x, mask):
        attn = lambda x : self.self_attn(attn_from=x, attn_to=x, value=x, mask=mask)
        
        x = self.resconnect[0](x, sublayer=attn)
        x = self.resconnect[1](x, sublayer=self.feed_fwd)
        return x
