import math

import torch
import torch.nn as nn


class DownsampleStridedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

        self.main = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2
            ),
            nn.LeakyReLU(self.negative_slope, inplace=True)
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
        x = self.main(x)

        return x


class DownsampleMaxPoolConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1
            ),
            nn.LeakyReLU(self.negative_slope, inplace=True)
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
        x = self.main(x)

        return x
