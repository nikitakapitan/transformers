"""
DecoderLayer implements 3 parts:
1. Masked multi-head self-attention + normalization + residual connection
2. Multi-head self-attention + normalization + residual connection
3. position-wise fully connected feed-forward + normalization + residual connection
"""
import torch.nn as nn
from transformers.helper import clones
from transformers.LayerNorm import LayerNorm
from transformers.ResidualConnection import ResidualConnection

class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn # MultiHeadedAttention(h, d_model)
        self.src_attn = src_attn   # MultiHeadedAttention(h, d_model)
        self.feed_fwd = feed_forward # PositionWiseFeedForward(d_model, d_ff)
        self.resconnect = clones(ResidualConnection(size, dropout), 3)
        # note : LayerNorm and DropOut are applied within 'resconnect'
        
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        tgt_attn = lambda x : self.self_attn(attn_from=x, attn_to=x, value=x, mask=tgt_mask)
        src_attn = lambda x : self.src_attn(attn_from=x, attn_to=m, value=m, mask=src_mask)
        
        x = self.resconnect[0](x, sublayer=tgt_attn)
        x = self.resconnect[1](x, sublayer=src_attn)
        x = self.resconnect[2](x, sublayer=self.feed_fwd)
        
        return x
