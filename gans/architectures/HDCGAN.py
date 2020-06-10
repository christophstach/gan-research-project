import torch.nn as nn
import torch.nn.functional as F

import gans.building_blocks as bb


class FirstHDCGANBlock(nn.Module):
    def __init__(self, noise_size, filters, bias=False):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            noise_size,
            filters,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            filters,
            filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.norm = bb.PixelNorm()

    def forward(self, x):
        x = self.norm(x)    

        x = self.conv1(x)
        F.selu(x, inplace=True)
        x = self.norm(x)

        x = self.conv2(x)
        F.selu(x, inplace=True)
        x = self.norm(x)

        return x


class UpsampleHDCGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.norm = bb.PixelNorm()

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(
                x.size(2) // 2,
                x.size(3) // 2
            ),
            mode="bilinear",
            align_corners=False
        )

        x = self.conv1(x)
        F.selu(x, inplace=True)
        x = self.norm(x)

        x = self.conv2(x)
        F.selu(x, inplace=True)
        x = self.norm(x)

        return x


class LastHDCGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, additional_channels, bias=False):
        super().__init__()

        self.miniBatchStdDev = bb.MinibatchStdDev()

        self.conv1 = nn.Conv2d(
            in_channels + additional_channels + 1,
            in_channels + additional_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels + additional_channels,
            out_channels + additional_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=bias
        )

        self.validator = nn.Conv2d(
            out_channels + additional_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.miniBatchStdDev(x)
        x = self.conv1(x)
        F.selu(x, inplace=True)
        x = self.conv2(x)
        F.selu(x, inplace=True)
        x = self.validator(x)

        return x


class DownsampleHDCGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.avgPool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        F.selu(x, inplace=True)
        x = self.conv2(x)
        F.selu(x, inplace=True)

        x = self.avgPool(x)

        return x
