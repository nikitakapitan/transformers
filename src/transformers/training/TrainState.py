class TrainState:
    "Track number of steps, examples and tokens processed"

    step: int = 0         # steps in the current epoch
    accum_step: int = 0   # nb of gradient accum steps
    samples: int = 0      # total nb of examples used
    tokens: int = 0       # total nb of tokens processed