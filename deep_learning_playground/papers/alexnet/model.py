import torch
from torch import nn
from deep_learning_playground.normalization import LocalResponseNormalization
from deep_learning_playground.regularization import Dropout
from deep_learning_playground.activation import ReLU


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

        self.convBase = nn.Sequential(self.firstConv,
                                      self.secondConv,
                                      self.thirdConv,
                                      self.fourthConv,
                                      self.fifthConv)

        self.firstFC = nn.Sequential
        (
            nn.Linear(256 * 6 * 6, 4096),
            Dropout(p=),

        )

    def forward(self, x):
        x = self.convBase(x)
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
