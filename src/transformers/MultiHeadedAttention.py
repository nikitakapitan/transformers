
from turtle import forward
from torch import dropout
import torch.nn as nn
from helper import clones
from attention import attention

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, p_dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.q_fc, self.k_fc, self.v_fc = clones(nn.Linear(d_model, d_model), 3)
        self.final_fc = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # same mask appleid to all h heads
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        query, key, value = [
                lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                for lin, x in zip((self.q_fc, self.k_fc, self.v_fc), (query, key, value))
            ]
        # apply attention on all projected vectors in batch
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)

        # Concat heads into multi-heads
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        del query, key, value
        return self.final_fc(x)
        
        