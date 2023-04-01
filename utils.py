import torch

def moving_average(x, n=10):
    avg = x.cumsum(dim=0)
    avg[n:] = avg[n:] - avg[:-n]
    return avg

def generate_sequence(batch_size, size, n):
    seq = (torch.rand((batch_size, size, n)) > 0.5).float()
    seq[:, :, -1] = 0.5
    return seq
