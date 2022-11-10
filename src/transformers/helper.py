
import torch
from copy import deepcopy

def following_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    following_mask = torch.tril(torch.ones(attn_shape)).type(torch.uint8)
    return following_mask

def clones(module, N):
    return torch.nn.ModuleList([deepcopy(module) for _ in range(N)])


