import torch
from torch import nn
from utils import LocalResponseNormalization, ReLU


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # Most network parameters are given in the paper, but padding has to
        # be inferred from dimensions given in Figure 2
        self.firstConv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            ReLU(),
            LocalResponseNormalization(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.secondConv = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            ReLU(),
            LocalResponseNormalization(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.thirdConv = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        self.fourthConv = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            ReLU(),
        )
        self.fifthConv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.secondConv(x)
        x = self.thirdConv(x)
        x = self.fourthConv(x)
        x = self.fifthConv(x)
        return x


if __name__ == '__main__':
    # test = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
    #                      nn.MaxPool2d(kernel_size=3, stride=2),
    #                      nn.Conv2d(96, 256, kernel_size=5, stride=1,
    #                                padding=2),
    #                      nn.MaxPool2d(kernel_size=3, stride=2),
    #                      nn.Conv2d(256, 384, kernel_size=3, stride=1,
    #                                padding=1),
    #                      nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
    #                      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    #                      )
    test = AlexNet()
    tensor = torch.rand((1, 3, 224, 224))
    print(test(tensor).shape)
