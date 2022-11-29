from os.path import exists
import torch
from transformers.training.train import train_model
from transformers.main import make_model

# def load_trained_model(src_vocab_len, tgt_vocab_len):
    
#     config = {
#         'batch_size' : 32,
#         'distributed' : False,
#         'num_epochs' : 8,
#         'accum_iter' : 10,
#         'base_lr' : 1.0,
#         'max_padding' : 72,
#         'warmup' : 3000,
#         'file_prefix' : 'multi30k_model_',
#     }
#     architecture = {
#         'src_vocab_len' : src_vocab_len,
#         'tgt_vocab_len' : tgt_vocab_len,
#         'N' : 6, # loop
#         'd_model' : 512, # emb
#         'd_ff' : 2048,
#         'h' : 8,
#         'dropout' : 0.1
#     }
#     model_path = 'multi30k_model_final.pt'
    

#     model = make_model(
#         src_vocab_len=architecture['src_vocab_len'],
#         tgt_vocab_len=architecture['tgt_vocab_len'],
#         N=architecture['N'],
#         d_model=architecture['d_model'],
#         d_ff=architecture['d_ff'],
#         h=architecture['h'],
#         dropout=architecture['dropout']
#     )
#     model.load_state_dict(torch.load(model_path))
#     return model