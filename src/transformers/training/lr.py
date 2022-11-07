

def rate(step, model_size, factor, warmup):

    if step == 0:
        step = 1
    
    floored_step = min(step ** (-0.5), step * warmup ** (-1.5))
    return factor * (model_size ** (-0.5) * floored_step)