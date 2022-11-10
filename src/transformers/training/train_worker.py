import torch
import torch.nn as nn
from transformers.main import make_model
from transformers.training.LabelSmoothing import LabelSmoothing
from transformers.data.load import create_dataloaders
from transformers.training.lr import rate
from transformers.training.TrainState import TrainState
from transformers.training.run_epoch import run_epoch
from transformers.data.Batch import Batch
from transformers.training.SimpleLossCompute import SimpleLossCompute
from transformers.helper import DummyOptimizer, DummyScheduler

import GPUtil

def train_worker(gpu, ngpus_per_node, vocab_src, vocab_tgt,
    spacy_de, spacy_en, config, is_distributed=False):
    """
    define train criterion
    import data loaders (train, valid)
    define optimizer
    define lr_scheduler
    init TrainState()
    """

    print(f'Train worker process using GPU :{gpu}', flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True

    criterion = LabelSmoothing(size=len(vocab_tgt),
     padding_idx=pad_idx, smoothing=0.1)
    # criterion = nn.KLDivLoss(reduction="sum")
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        device=gpu,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        spacy_de=spacy_de,
        spacy_en=spacy_en,
        batch_size=config['batch_size'] // ngpus_per_node,
        max_padding=config['max_padding'],
        is_distributed=False
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config['base_lr'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step=step, model_size=d_model, factor=1, warmup=config['warmup']
        )
    )
    train_state = TrainState()

    for epoch in range(config['num_epochs']):

        model.train()
        print(f'[GPU{gpu}] Epoch {epoch} Training ====', flush=True)
        _, train_state = run_epoch(
            data_iter=(Batch(src=b[0], tgt=b[1], pad=pad_idx) for b in train_dataloader),
            model=model,
            loss_compute=SimpleLossCompute(generator=module.generator, criterion=criterion),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode='train+log',
            accum_iter=config['accum_iter'],
            train_state=train_state,
        )
        GPUtil.showUtilization()
        if is_main_process:
            file_path = f"{config['file_prefix']}.{epoch}.pt"
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ===", flush=True)
        model.eval()

        sloss = run_epoch(
            data_iter=(Batch(src=b[0], tgt=b[1], pad=pad_idx) for b in valid_dataloader),
            model=model,
            loss_compute=SimpleLossCompute(generator=module.generator, criterion=criterion),
            optimizer=DummyOptimizer(),
            scheduler=DummyScheduler(),
            mode='eval',
        )
        print(sloss)
        torch.cuda.empty_cache()
    
    if is_main_process:
        file_path = f"{config['file_prefix']}final.pt"
        torch.save(module.state_dict(), file_path)