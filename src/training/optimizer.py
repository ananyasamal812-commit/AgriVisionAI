import torch.optim as optim


def get_optimizer(model, lr=1e-3):

    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )