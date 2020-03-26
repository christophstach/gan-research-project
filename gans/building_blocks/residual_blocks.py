import math

import torch
import torch.nn as nn


class ResidualBlockTypeA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.last = nn.LeakyReLU(self.negative_slope, inplace=True)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity="leaky_relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.main(x)

        return self.last(identity + x)


class ResidualBlockTypeB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity="leaky_relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.main(x)

        return identity + x


class ResidualBlockTypeC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            )
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, a=self.negative_slope, nonlinearity="leaky_relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.main(x)

        return identity + x
