import torch
from torch import nn
from torch.autograd import Function

def PCC(a: torch.tensor, b: torch.tensor):
    am = torch.mean(a, dim=0)
    bm = torch.mean(b, dim=0)
    num = torch.sum((a - am) * (b - bm), dim=0)
    den = torch.sqrt(sum((a - am) ** 2) * sum((b - bm) ** 2))
    return num/den


def CCC(a: torch.tensor, b: torch.tensor):
    rho = 2 * ((a - a.mean(dim=0)) * (b - b.mean(dim=0))).mean(dim=0)
    rho /= a.var(dim=0) + b.var(dim=0) + torch.pow(a.mean(dim=0) - b.mean(dim=0), 2)
    return rho
