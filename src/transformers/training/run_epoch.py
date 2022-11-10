from transformers.training.TrainState import TrainState
import time

def run_epoch(data_iter, model, loss_compute, optimizer, scheduler,
                 mode='train', accum_iter=1, train_state=TrainState()):
    """Train a single epoch.
    data_iter
    model : ex. EncoderDecoder
    """
    start = time.time()
    total_tokens, total_loss, tokens, n_accum = 0, 0, 0, 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode =='train' or mode=='train+log':
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == 'train' or mode == 'train+log'):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(f"Epoch Step {i} | Accum Step : {n_accum} | lr={lr} |\
                 Loss {loss/batch.ntokens}\ | Tokens / Sec {tokens/elapsed} ")
            start = time.time()
            tokens = 0
            del loss
            del loss_node
    return total_loss / total_tokens, train_state