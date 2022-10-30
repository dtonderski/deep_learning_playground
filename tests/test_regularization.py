import torch
from deep_learning_playground.regularization import Dropout
import pytest

generator = torch.Generator()
generator.manual_seed(21309124)
shape = (10, 3, 224, 224)

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
