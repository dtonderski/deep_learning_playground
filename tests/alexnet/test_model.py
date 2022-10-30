import pytest
import torch
from deep_learning_playground.papers.alexnet.model import AlexNet

generator = torch.Generator()
generator.manual_seed(21309124)
shape = (10, 3, 224, 224)


def test_conv_shapes():
    x = torch.rand(shape, generator=generator)
    conv_output_shapes = ((96, 27, 27), (256, 13, 13), (384, 13, 13),
                          (384, 13, 13), (256, 6, 6))
    model = AlexNet()
    for layer, output_shape in zip(model.convBase, conv_output_shapes):
        x = layer(x)
        assert x.shape[1:] == output_shape


@pytest.mark.parametrize('n_outputs', [100, 1000])
def test_fc_shapes(n_outputs):
    x = torch.rand(shape, generator=generator)
    model = AlexNet(n_outputs)
    x = model.convBase(x)
    x = torch.flatten(x, 1)

    fc_output_shapes = (4096, 4096, n_outputs)
    for layer, output_shape in zip(model.classifier, fc_output_shapes):
        x = layer(x)
        assert x.shape[1] == output_shape
