import torch
from torch import nn


class LocalResponseNormalization(nn.Module):
    n: int
    alpha: float
    beta: float
    k: float

    def __init__(self, n=5, alpha=10 ** (-4), beta=0.75, k=2):
        super(LocalResponseNormalization, self).__init__()
        self.n, self.alpha, self.beta, self.k = n, alpha, beta, k

    def forward(self, tensor: torch.Tensor):
        divisor = tensor * tensor

        kernel_size = (self.n, 1, 1)
        padding = (self.n // 2, 0, 0)

        divisor = nn.functional.avg_pool3d(divisor, kernel_size, stride=1,
                                           padding=padding,
                                           count_include_pad=False)

        divisor = (self.k + self.alpha * divisor) ** self.beta

        return tensor / divisor
