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


def histogram(x: torch.tensor, bins: torch.tensor) -> torch.tensor:
    hist = torch.zeros_like(bins).cuda()
    for i in range(1, len(bins)):
        upper_lim = bins[i]
        lower_lim = bins[i - 1]
        hist[i - 1] = torch.sum(x < upper_lim, dim=0) - torch.sum(x < lower_lim, dim=0)
    hist /= x.size(0)
    return hist


def onehotprob(x: float, bins: torch.tensor, hist: torch.tensor) -> torch.tensor:
    for i in range(1, len(bins)):
        upper_lim = bins[i]
        lower_lim = bins[i - 1]
        if (x > lower_lim) & (x <= upper_lim):
            return hist[i]
    return 0


class OneHotProb(Function):
    @staticmethod
    def forward(ctx, x, bins):
        hist = histogram(x, bins)
        # ctx.save_for_backward(x)
        # ctx.save_for_backward(hist)
        out = torch.zeros_like(x).cuda()
        for idx, row in enumerate(x):
            for ydx, el in enumerate(row):
                out[idx, ydx] = onehotprob(el, hist[:, ydx], bins[:, ydx])
        ctx.save_for_backward(out)
        # print(x.shape, out.shape)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        out, = ctx.saved_tensors
        # print(grad_input.shape)
        # grad_input *= out
        return grad_input, None


class Histogram(Function):
    @staticmethod
    def forward(ctx, x, bins):
        hist = histogram(x, bins)
        ctx.save_for_backward(x)
        ctx.save_for_backward(hist)
        return hist
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        return grad_input, None
