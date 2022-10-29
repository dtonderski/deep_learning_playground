import torch
import torch.nn.functional as F
from deep_learning_playground.papers.alexnet.utils \
    import LocalResponseNormalization, ReLU

generator = torch.Generator()
generator.manual_seed(21309124)
shape = (10, 64, 224, 224)


def assert_normalization_correct(tensor, n, alpha, beta, k, tol, inplace):
    normalization = LocalResponseNormalization(n, alpha, beta, k, inplace)

    official_normalized = F.local_response_norm(tensor, n, alpha, beta, k)

    if inplace:
        homemade_normalized = tensor.clone()
        normalization(homemade_normalized)
    else:
        homemade_normalized = normalization(tensor)

    difference_abs = torch.abs(homemade_normalized - official_normalized)

    assert torch.all(difference_abs < tol)


def test_local_response_normalization():
    tol = 10 ** (-4)

    n, alpha, beta, k = 5, 10 ** (-4), 0.75, 2

    tensor = torch.rand(shape, generator=generator)

    assert_normalization_correct(tensor, n, alpha, beta, k, tol, inplace=False)
    assert_normalization_correct(tensor, n, alpha, beta, k, tol, inplace=True)


def assert_relu_correct(tensor, inplace):
    relu = ReLU(inplace)
    if inplace:
        output = tensor.clone()
        relu(output)
    else:
        output = relu(tensor)
    assert torch.all(torch.logical_or(tensor < 0, torch.eq(tensor, output)))


def test_relu():
    tensor = torch.rand(shape, generator=generator)
    assert_relu_correct(tensor, False)
    assert_relu_correct(tensor, True)
