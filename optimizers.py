from torch.optim import Adam, AdamW, SGD


def build_optimizer(optim_name, model_params, lr):
    if optim_name.lower() == "adam":
        optimizer = Adam(model_params, lr)
    elif optim_name.lower() == "adamw":
        optimizer = AdamW(model_params, lr)
    elif optim_name.lower() == "sgd":
        optimizer = SGD(model_params, lr)
    else:
        raise ValueError(f'Optimizer name {optim_name} is not valid.')

    return optimizer