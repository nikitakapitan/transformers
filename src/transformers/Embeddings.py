

import math
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_len, d_model) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_len, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
