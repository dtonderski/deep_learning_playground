import torchvision
import torch
import numpy as np
from collections import defaultdict


class ReverseConv2d(torch.nn.ConvTranspose2d):
    """
    This class is a reverse of Conv2d - it takes in a Conv2d and returns
    the reverse using ConvTranspose2d with 180 rotated conv2d weights.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        super().__init__(conv2d.out_channels, conv2d.in_channels,
                         conv2d.kernel_size, conv2d.stride, conv2d.padding)
        with torch.no_grad():
            self.weight.data = torch.tensor(np.rot90(conv2d.weight, k=2,
                                                     axes=(2, 3)).copy())
            self.bias.data[:] = 0


class ReverseMaxPool2d(torch.nn.Module):
    """
    This class is a reverse of MaxPool2d. See https://arxiv.org/abs/1311.2901
    for details.
    """
    layer_index: int
    switches: dict
    kernel_size: int
    stride: int
    pre_pool_shapes: dict

    def __init__(self, maxPool2d: torch.nn.MaxPool2d, switches,
                 layer_index, pre_pool_shapes):
        """

        :param maxPool2d: the pooling layer to reverse
        :param switches: a dict mapping layer indices to the switches for the
                         pooling in that layer
        :param layer_index: layer index of the pooling that we are reversing
        :param pre_pool_shapes: shapes of the feature maps before pooling.
                                This is the output shape of this class.
        """
        super().__init__()
        self.switches = switches
        self.layer_index = layer_index
        self.kernel_size, self.stride = maxPool2d.kernel_size, maxPool2d.stride
        self.pre_pool_shapes = pre_pool_shapes

    def forward(self, x):
        output_shape = self.pre_pool_shapes[self.layer_index]
        indices = self.switches[self.layer_index].detach()

        # Indices is a tensor mapping each element in x to the index that it
        # got pooled from. The plan is to unravel all those indices. However,
        # that will only give us width and height, we also need channel, which
        # we get below.
        new_indices = torch.zeros_like(indices)
        for i in range(new_indices.shape[1]):
            new_indices[0, i, :] = i

        unpooled_features = torch.zeros(output_shape)
        w, h = output_shape[2:4]
        unpooled_features[0, new_indices, np.unravel_index(indices, (w, h))[0],
                          np.unravel_index(indices, (w, h))[1]] = x
        return unpooled_features


class AlexNetWrapper(torchvision.models.AlexNet):
    """
    This class wraps AlexNet, saving the feature maps after each Conv2d layer
    on each run, and saving the pooling indices after each MaxPool2d layer
    """

    def __init__(self,
                 weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1,
                 progress: bool = True):
        super().__init__()
        if weights is not None:
            super().load_state_dict(weights.get_state_dict(progress=progress))

        self.layers_dict = defaultdict(list)
        self.feature_maps = dict()
        self.switches = dict()
        self.reversed_layers_dict = defaultdict(list)
        self.pre_pool_shapes = dict()
        self.max_pools = list()

        # Rewrite alexNet.features so that we can save switches for max pool
        # and feature maps for conv layers from last run
        i_layer = -1

        for sublayer in self.features:
            if type(sublayer) == torch.nn.Conv2d:
                i_layer += 1
                self.reversed_layers_dict[i_layer].insert(0, ReverseConv2d(
                    sublayer))
            elif type(sublayer) == torch.nn.MaxPool2d:
                self.max_pools.append(sublayer)
                sublayer.return_indices = True
                self.reversed_layers_dict[i_layer].insert(0, ReverseMaxPool2d(
                    sublayer, self.switches, i_layer, self.pre_pool_shapes))
            else:
                # Reverted ReLU is just ReLU
                self.reversed_layers_dict[i_layer].insert(0, torch.nn.ReLU())

            self.layers_dict[i_layer] += [sublayer]

    def change_return_indices(self, mode=True):
        for max_pool in self.max_pools:
            self.feature_maps.clear()
            self.switches.clear()

            max_pool.return_indices = mode

    def forward(self, x, original_mode=False):
        if original_mode:
            self.change_return_indices(False)
            return super().forward(x)
        self.change_return_indices(True)
        for i_layer, layers in self.layers_dict.items():
            for layer in layers:
                if type(layer) == torch.nn.Conv2d:
                    x = layer(x)
                    self.feature_maps[i_layer] = x.clone()
                elif type(layer) == torch.nn.MaxPool2d:
                    self.pre_pool_shapes[i_layer] = x.shape
                    x, indices = layer(x)
                    self.switches[i_layer] = indices
                else:
                    x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def reverse_feature_map(self, start_layer, i_feature_map, stop_layer=0):
        x = self.feature_maps[start_layer]
        x[:, i_feature_map, :] = 0
        initial_layer = self.reversed_layers_dict[start_layer][-1]
        x = initial_layer(x)

        if start_layer > 0:
            if stop_layer == 0:
                layers_to_iterate = list(self.reversed_layers_dict.items())[
                                    start_layer - 1::-1]
            else:
                layers_to_iterate = list(self.reversed_layers_dict.items())[
                                    start_layer - 1:stop_layer:-1]

            for i_layer, layers in layers_to_iterate:
                for layer in layers:
                    x = layer(x)
        return x
