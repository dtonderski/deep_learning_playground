import torch
from torch import nn

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, tensor: torch.Tensor):
        return torch.maximum(tensor, torch.tensor(0))