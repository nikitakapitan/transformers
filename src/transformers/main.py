
import torch.nn as nn
from copy import deepcopy as dcopy
from MultiHeadedAttention import MultiHeadedAttention
from PositionWiseFeedForward import PositionWiseFeedForward

from PositionalEncoding import PositionalEncoding
from Embeddings import Embeddings

from EncoderDecoder import EncoderDecoder
from Encoder import Encoder
from EncoderLayer import EncoderLayer
from Decoder import Decoder
from DecoderLayer import DecoderLayer

from Generator import Generator

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(
            layer=EncoderLayer(
                size=d_model,
                self_attn=dcopy(attn),
                feed_fwd=dcopy(ff),
                dropout=dropout
            ),
            N=N #6
        ),
        decoder=Decoder(
            layer=DecoderLayer(
                size=d_model,
                self_attn=dcopy(attn),
                src_attn=dcopy(attn),
                feed_fwd=dcopy(ff),
                dropout=dropout
            ),
            N=N #6
        ),
        src_emb=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab=src_vocab
            ),
            dcopy(position)
        ),
        tgt_emb=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab=tgt_vocab
            ),
            dcopy(position)
        ),
        generator=Generator(
            d_model=d_model,
            vocab=tgt_vocab
        )

    )

    # init params with Glorot / fan_avg:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model