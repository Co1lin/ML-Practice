import torch
from torch import nn

class ResNet(nn.Module):

    def __init__(self, num_classes, channels=[64, 64, 128, 256, 512], repeats=[0, 2, 3, 5, 2]):
        """
        """
        super().__init__()
        if len(channels) != len(repeats):
            raise ValueError(f'len(channels) = {len(channels)} is not equal to len(repeats) = {len(repeats)}')

        self.conv1 = nn.Sequential(
            # (B, 3, W, H) -> (B, channels[0], W/2, H/2)
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]),
            nn.SELU(),
        )
        convs = []
        # (B, channels[0], W/2, H/2) -> (B, channels[1], W/4, H/4)
        convs.append(ResBlock(channels[0], channels[1], repeats[1], max_pool=True))
        for i in range(2, len(channels)):
            convs.append(ResBlock(channels[i - 1], channels[i], repeats[i]))

        self.conv_layers = nn.Sequential(*convs)

        self.classifier = nn.Sequential(
            # (B, C, W, H) -> (B, C, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            # (B, C, 1, 1) -> (B, C)
            nn.Flatten(1),
            # (B, C) -> (B, num_classes)
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, img):
        """

        :param img: (B, 3, W, H)
        :return:
        """
        # (B, 3, W, H) -> (B, 64, W/2, H/2)
        out = self.conv1(img)
        out = self.conv_layers(out)
        out = self.classifier(out)
        return out


class ResBlock(nn.Module):

    def __init__(self, last_channels, this_channels, repeat_time, max_pool = False):
        """
        (B, C_last, W, H) -> (B, C_this, W/2, H/2)
        """
        super().__init__()

        self.repeat_time = repeat_time
        self.activate = nn.SELU()
        self.bn = nn.BatchNorm2d
        self.max_pool = None

        if max_pool:
            self.max_pool = nn.MaxPool2d(3, 2, 1)
        else:
            self.conv1_1 = nn.Sequential(
                nn.Conv2d(last_channels, this_channels, 3, stride=2, padding=1),
                self.bn(this_channels),
            )
            self.conv1_2 = nn.Sequential(
                nn.Conv2d(this_channels, this_channels, 3, stride=1, padding=1),
                self.bn(this_channels),
            )

        self.conv2 = nn.Sequential(
            nn.Conv2d(this_channels, this_channels, 3, stride=1, padding=1),
            self.bn(this_channels),
            self.activate,
            nn.Conv2d(this_channels, this_channels, 3, stride=1, padding=1),
            self.bn(this_channels),
        )

    def forward(self, x):
        """
        :param x: (B, C_last, W, H)
        :return: (B, C_this, W/2, H/2)
        """
        if self.max_pool is not None:
            out = self.max_pool(x)
        else:
            # downsample
            identity = self.conv1_1(x)
            # (B, C_last, W, H) -> (B, C_this, W/2, H/2)
            out = self.conv1_1(x)
            out = self.activate(out)
            out = self.conv1_2(out)
            out += identity
            out = self.activate(out)


        for _ in range(self.repeat_time):
            identity = out
            out = self.conv2(out)
            out += identity
            out = self.activate(out)

        return out