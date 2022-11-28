

import math
import torch


def attention(attn_from, attn_to, value, mask=None, dropout=None):

    d_k = attn_from.size(-1)
    scores = torch.matmul(attn_from, attn_to.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    context = torch.matmul(p_attn, value)
    return context, p_attn