import torch

def BradleyTerry_loss(probs):
    probs = -torch.log(probs)
    loss = torch.sum(probs)
    return loss
