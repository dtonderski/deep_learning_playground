import torch
from torch import nn


class Dropout(nn.Module):
    p: float

    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        mask = torch.rand(x.shape)
        mask = mask > self.p
        return x.mul(mask)
