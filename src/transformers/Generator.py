
import torch.nn as nn
from torch.nn.functional import log_softmax

class Generator(nn.Module):

    def __init__(self, d_model, vocab_len):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_len)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
