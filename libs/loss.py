import torch

def LossMSE(target, predicted):
    return ((target - predicted)**2).mean()