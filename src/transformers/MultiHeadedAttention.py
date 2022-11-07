

from torch import dropout
import torch.nn as nn
from transformers.helper import clones
from transformers.attention import attention

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, p_dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_head = d_model // h
        self.h = h
        self.q_fc, self.k_fc, self.v_fc = clones(nn.Linear(d_model, d_model), 3)
        self.final_fc = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, attn_from, attn_to, value, mask=None):
        "query, key, value : tensor (n_batch, n_tokens, d_model)"
        if mask is not None:
            # same mask appleid to all h heads
            mask = mask.unsqueeze(1) # *(1, 
        n_batches = value.size(0)
        
        # Compute Query, Key & Value. shape -> (n_batch, n_tokes, d_model)
        query = self.q_fc(attn_from) 
        key = self.k_fc(attn_to)
        value = self.v_fc(value)

        # review shape -> (n_batches, n_heads, n_tokens, d_head) 
        query = query.view(n_batches, -1, self.h, self.d_head).transpose(1, 2)
        key = key.view(n_batches, -1, self.h, self.d_head).transpose(1, 2)
        value = value.view(n_batches, -1, self.h, self.d_head).transpose(1, 2)

        # apply attention on all projected vectors in batch
        context, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # Concat heads into multi-heads
        context = context.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_head)
        del query, key, value
        return self.final_fc(context)
        
        