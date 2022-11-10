from transformers.data.token import tokenize
from transformers.data.Batch import collate_batch

import torchtext.datasets as datasets
from torchtext.data.functional import to_map_style_dataset

from torch.utils.data import DataLoader

def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en,
                     batch_size=8000, max_padding=128):

    def tokenize_de(text):
        return tokenize(text, spacy_de)
    
    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch=batch,
            src_pipeline=tokenize_de,
            tgt_pipeline=tokenize_en,
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    # dataset from iterable to map
    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = None
    
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader