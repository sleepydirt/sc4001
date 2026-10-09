import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1):
    # SAME padding. PyTorch rejects padding="same" when stride > 1, so pad explicitly.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            conv_bn_relu(1, 16, 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn_relu(16, 32, 5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn_relu(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.linear_stack = nn.Sequential(
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.linear_stack(x)


class VGG(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            conv_bn_relu(1, 8, 3),
            conv_bn_relu(8, 8, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn_relu(8, 16, 3),
            conv_bn_relu(16, 16, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_bn_relu(16, 32, 3),
            conv_bn_relu(32, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, projection=False):
        super().__init__()
        self.main_path = nn.Sequential(
            conv_bn_relu(in_channels, out_channels, 3, stride=stride),
            # ReLU comes after the addition
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            conv_bn_relu(in_channels, out_channels, 1, stride=stride)
            if projection
            else nn.Identity()
        )

    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu(1, 16, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(16, 16, stride=1, projection=False),
            ResidualBlock(16, 32, stride=2, projection=True),
            ResidualBlock(32, 64, stride=2, projection=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
