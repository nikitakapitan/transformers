"""
This function is a fancy capsula
The core function is transformers.training.train_worker
"""
from transformers.training.train_worker import train_worker

def train(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    train_worker(
        gpu=0,
        ngpus_per_node=1,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        config=config,
        is_distributed=False,
    )

def load_trained_model():
    config = {
        'batch_size' : 32,
        'distributed' : False,
        'num_epochs' : 8,
        'accum_iter' : 10,
        'base_lr' : 1.0,
        'max_padding' : 72,
        'warmup' : 3000,
        'file_prefix' : 'multi30k_model_',
    }
    model_path = 'multi30k_model_final.pt'
    
   
