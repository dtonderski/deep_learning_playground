import pytest

import torch
from torch import nn
from deep_learning_playground.papers.alexnet.utils import \
    LocalResponseNormalization, ReLU, Dropout

generator = torch.Generator()
generator.manual_seed(21309124)
# Can't have 3 channels as we have 5 kernels in normalization
shape = (10, 10, 224, 224)


def test_relu():
    tensor = torch.rand(shape, generator=generator)
    relu = ReLU()
    output = relu(tensor)
    assert torch.all(torch.logical_or(tensor < 0, torch.eq(tensor, output)))


def test_local_response_normalization():
    tol = 10 ** (-4)
    n, alpha, beta, k = 5, 10 ** (-4), 0.75, 2

    tensor = torch.rand(shape, generator=generator)
    normalization = LocalResponseNormalization(n, alpha, beta, k)

    official_normalized = nn.functional.local_response_norm(tensor, n, alpha,
                                                            beta, k)

    difference_abs = torch.abs(normalization(tensor) - official_normalized)

    assert torch.all(difference_abs < tol)


p_data = [0, 0.25, 0.5, 0.75, 1]


@pytest.mark.parametrize("p", p_data)
def test_dropout(p):
    # This test will always be intrinsically probabilistic. However, with
    # a tolerance of 0.01 and 10*3*224*224 samples, the probability of
    # incorrectly failing the test is very low
    tol = 0.01
    dropout = Dropout(p)
    x = torch.rand(shape, generator=generator)
    x = dropout(x)
    frac_zeros = (x == 0).sum() / (x.numel())
    assert torch.abs(frac_zeros - p) < tol
