from transformers.helper import following_mask

# import for def collate_batch
import torch
from torch.nn.functional import pad

class Batch:
    "Goal: hold batch of data with mask during training"

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    
    @staticmethod
    def make_mask(tgt, pad):
        "create a mask to hide padding and future words"
        hide_padding = (tgt != pad).unsqueeze(-2) 
        hide_future_words = following_mask(tgt.size(-1)).type_as(hide_padding.data)

        return hide_padding & hide_future_words


def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab,
                  tgt_vocab, device, max_padding=128, pad_id=2):
    """
    Given a batch of (src, tgt):
    1. wrap src and tgt with <s> & </s>
    2. add padding with value=pad_id 
    3. stack, creating new dimension

    batch : 
    src_pipeline : 
    tgt_pipeline : 
    """
    bs_id = torch.tensor([0], device=device) #<s> token id
    eos_id = torch.tensor([1], device=device) #</s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        # wrap src with <s> & </s>
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            dim=0,
        )
        # wrap tgt with <s> & </s>
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            dim=0,
        )
        # wrap src with pad_id from right
        src_list.append(
            pad(
                input=processed_src,
                # 0 from left and rest from right
                pad=(0, max_padding - len(processed_src)),
                value=pad_id, # pad value
            )
        )
        # wrap tgt with pad_id from right
        tgt_list.append(
            pad(
                input=processed_tgt,
                pad=(0, max_padding - len(processed_tgt)),
                value = pad_id,
            )
        )

        # concat creating new dimension
        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return (src, tgt)
