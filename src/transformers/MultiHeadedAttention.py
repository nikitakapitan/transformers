

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
        """ 
        Attention is applied within last dimension slice by slice. 
        attn_from, attn_to, value might have any arbitrary dimension.

        In decoder case, attn_from dimension is dynamic (from 1 to N)
        
        If different dimensions : q_fc, k_fc, v_fc are responsible to convert arbitrary input to fixed dimension.

        query :  (n_batch, n_tokens_from, d_model)
        key :  (n_batch, n_tokens_to, d_model)
        value :  (n_batch, n_tokens_value, d_model)
        """

        if mask is not None:
            # same mask appleid to all h heads
            mask = mask.unsqueeze(1) # *(1, 
        n_batches = value.size(0)

        n_tokens_from = attn_from.size(1)
        n_tokens_to = attn_to.size(1)
        n_tokens_value = value.size(1)
        
        # Compute Query, Key & Value. shape -> (n_batch, n_tokes, d_model)
        query = self.q_fc(attn_from) 
        key = self.k_fc(attn_to)
        value = self.v_fc(value)

        # Split to H heads
        # |view| -> (n_batches, n_tokens, self.h, self.d_head)
        # |transpose| -> (n_batches, n_heads, n_tokens, d_head) 
        query = query.view(n_batches, n_tokens_from, self.h, self.d_head).transpose(1, 2)
        key = key.view(n_batches, n_tokens_to, self.h, self.d_head).transpose(1, 2)
        value = value.view(n_batches, n_tokens_value, self.h, self.d_head).transpose(1, 2)

        # attention : context -> (n_batches, n_heads, n_tokens_from, d_h)
        context, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Collapse heads. transpose -> (n_batches, n_tokens_from, n_heads, d_h). view -> (n_batches, n_tokens_from, d_model)
        context = context.transpose(1, 2).contiguous().view(n_batches, n_tokens_from, self.h * self.d_head)
        del query, key, value
        return self.final_fc(context)
        
        
