import torch
from deep_learning_playground.activation import ReLU

generator = torch.Generator()
generator.manual_seed(21309124)
shape = (10, 3, 224, 224)


def test_relu():
    tensor = torch.rand(shape, generator=generator)
    relu = ReLU()
    output = relu(tensor)
    assert torch.all(torch.logical_or(tensor < 0, torch.eq(tensor, output)))
