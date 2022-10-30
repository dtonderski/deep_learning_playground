import torch
from torch.nn import functional
from deep_learning_playground.normalization import LocalResponseNormalization

generator = torch.Generator()
generator.manual_seed(21309124)
shape = (10, 3, 224, 224)

def test_local_response_normalization():
    tol = 10 ** (-4)
    n, alpha, beta, k = 5, 10 ** (-4), 0.75, 2

    tensor = torch.rand(shape, generator=generator)
    normalization = LocalResponseNormalization(n, alpha, beta, k)

    official_normalized = functional.local_response_norm(tensor, n, alpha,
                                                         beta, k)

    difference_abs = torch.abs(normalization(tensor) - official_normalized)

    assert torch.all(difference_abs < tol)