from transformers.data.dataloader import create_dataloaders
import torch
from transformers.main import make_model
from transformers.data.Batch import Batch
from transformers.helper import following_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    tgt = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1): # loop over output words (decoded)
        tgt_mask = following_mask(tgt.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, tgt, tgt_mask)

        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1).unsqueeze(0)
        tgt=torch.cat([tgt, next_word],dim=1)

    return tgt

def check_outputs(valid_dataloader, model, vocab_src, vocab_tgt,
                n_examples=15, pad_idx=2, eos_string="</s>"):
    results = [()] * n_examples
    for idx in range(n_examples):
        print(f"\nExample {idx} ======\n")
        b = next(iter(valid_dataloader))
        rb = Batch(src=b[0], tgt=b[1], pad=pad_idx)
        # greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x!=pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x!=pad_idx]

        print(f"Source text (Input) {src_tokens}")
        print(f"Target Text (Ground Truth) {tgt_tokens}")

        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (" ".join([vocab_tgt.get_itos()[x] for x in model_out if x!= pad_idx])\
            .split(eos_string, 1)[0] + eos_string)
        print(f"Model Output {model_txt}")

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, architecture, n_examples=5):

    print('Preparing Data...')
    _, valid_dataloader = create_dataloaders(
        device=torch.device("cpu"),
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("checking Model Outputs:")
    example_data = check_outputs(valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples)
    return model, example_data
