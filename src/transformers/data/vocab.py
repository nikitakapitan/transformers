from transformers.data.token import tokenize, yield_tokens
from os.path import exists

import torch
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator

def build_vocab(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)
    
    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German vocab ...")

    train, val, test = datasets.Multi30k(language_pair=("de","en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train +  val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English vocab ...")
    train, val, test = datasets.Multi30k(language_pair=('de', 'en'))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        )
    
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

def load_vocab(spacy_de, spacy_en):
    if not exists('vocab.pt'):
        vocab_src, vocab_tgt = build_vocab(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    # print(f"DOne.\nVocab len: SRC={len(vocab_src)} TGT={len(vocab_tgt)}")
    return vocab_src, vocab_tgt