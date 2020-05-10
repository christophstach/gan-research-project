import torch.nn as nn
import torch.nn.functional as F

import gans.building_blocks as bb


class FirstHDCGANBlock(nn.Module):
    def __init__(self, noise_size, filters, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.block = nn.Sequential(
            # input is Z, going into a convolution

            bb.ConvTranspose2d(
                noise_size,
                filters,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(filters),
            nn.SELU(inplace=True),
            bb.Conv2d(
                filters,
                filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(filters),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)

        return x


class UpsampleHDCGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False, position=None):
        super().__init__()


        self.conv1 = nn.Sequential(
            bb.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            bb.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(
                x.size(2) * 2,
                x.size(3) * 2
            ),
            mode="bilinear",
            align_corners=False
        )

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class LastHDCGANBlock(nn.Module):
    def __init__(self, filters, additional_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.block = nn.Sequential(
            bb.MinibatchStdDev(),
            bb.Conv2d(
                filters // 2 + additional_channels + 1,
                filters + additional_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(filters + additional_channels),
            nn.SELU(inplace=True),
            bb.Conv2d(
                filters + additional_channels,
                filters + additional_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(filters + additional_channels),
            nn.SELU(inplace=True),
            bb.Conv2d(
                filters + additional_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            )
        )

    def forward(self, x):
        x = self.block(x)
        return x


class DownsampleHDCGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False, position=None):
        super().__init__()

        if position == 0:
            self.conv1 = nn.Sequential(
                bb.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    eq_lr=eq_lr,
                    spectral_normalization=spectral_normalization
                ),
                nn.SELU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                bb.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    eq_lr=eq_lr,
                    spectral_normalization=spectral_normalization
                ),
                nn.BatchNorm2d(in_channels),
                nn.SELU(inplace=True)
            )

        self.conv2 = nn.Sequential(
            bb.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=spectral_normalization
            ),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = F.interpolate(
            x,
            size=(
                x.size(2) // 2,
                x.size(3) // 2
            ),
            mode="bilinear",
            align_corners=False
        )

        return x
