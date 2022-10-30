import torch
from torch import nn


class LocalResponseNormalization(nn.Module):
    n: int
    alpha: float
    beta: float
    k: float
    inplace: bool

    def __init__(self, n=5, alpha=10 ** (-4), beta=0.75, k=2, inplace=False):
        super(LocalResponseNormalization, self).__init__()
        self.n, self.alpha, self.beta, self.k = n, alpha, beta, k
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor):
        divisor = tensor * tensor

        kernel_size = (self.n, 1, 1)
        padding = (self.n // 2, 0, 0)

        divisor = nn.functional.avg_pool3d(divisor, kernel_size, stride=1,
                                           padding=padding,
                                           count_include_pad=False)

        divisor = (self.k + self.alpha * divisor) ** self.beta
        if self.inplace:
            torch.div(tensor, divisor, out=tensor)
        else:
            return torch.div(tensor, divisor)
