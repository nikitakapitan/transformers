from transformers.helper import following_mask

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