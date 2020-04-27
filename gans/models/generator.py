import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gans.building_blocks as bb


class UpsampleSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = bb.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.leaky_relu(x)
        # x = self.pixelNorm(x)

        return x

class UpsampleProGANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = bb.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )
        self.conv2 = bb.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )
        self.pixelNorm = bb.PixelNorm()

    def forward(self, x):
        x = self.upsample(x)

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.pixelNorm(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.pixelNorm(x)

        return x


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_skip = bb.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        self.conv1 = bb.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        self.conv2 = bb.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        self.pixelNorm = bb.PixelNorm()

    def forward(self, x):
        x = self.upsample(x)
        x = self.pixelNorm(F.leaky_relu(self.conv_skip(x), 0.2))

        identity = x
        x = self.pixelNorm(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pixelNorm(F.leaky_relu(self.conv2(x), 0.2))

        return x + identity


class UpsampleSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = bb.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        self.att = bb.SelfAttention2d(out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)

    def forward(self, x):
        x = self.upsample(x)
        x = F.leaky_relu(self.conv(x), 0.2)
        x = F.leaky_relu(self.att(x), 0.2)

        return x


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.bias = True

        self.blocks = nn.ModuleList()
        self.to_rgb_converts = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                # input is Z, going into a convolution
                bb.PixelNorm(),
                
                bb.ConvTranspose2d(
                    self.hparams.noise_size,
                    self.hparams.generator_filters,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=self.bias,
                    eq_lr=self.hparams.equalized_learning_rate,
                    spectral_normalization=self.hparams.spectral_normalization
                ),
                nn.LeakyReLU(2.0, inplace=True),
                bb.PixelNorm(),

                bb.Conv2d(
                    self.hparams.generator_filters,
                    self.hparams.generator_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.bias,
                    eq_lr=self.equalized_learning_rate,
                    spectral_normalization=self.spectral_normalization
                ),
                nn.LeakyReLU(2.0, inplace=True),
                bb.PixelNorm()
            )
        )

        for i in range(2, int(math.log2(self.hparams.image_size))):
            self.blocks.append(
                self.block_fn(
                    self.hparams.generator_filters // 2 ** (i - 2),
                    self.hparams.generator_filters // 2 ** (i - 1),
                    self.bias,
                    self.hparams.equalized_learning_rate,
                    self.hparams.spectral_normalization
                )
            )

        for i in range(1, int(math.log2(self.hparams.image_size))):
            self.to_rgb_converts.append(
                self.to_rgb_fn(
                    self.hparams.generator_filters // 2 ** (i - 1),
                    self.bias,
                    self.hparams.equalized_learning_rate,
                    self.hparams.spectral_normalization
                )
            )

    def block_fn(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        # return UpsampleSelfAttentionBlock(in_channels, out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        # return UpsampleResidualBlock(in_channels, out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        # return UpsampleSimpleBlock(in_channels, out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        return UpsampleProGANBlock(in_channels, out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)

    def to_rgb_fn(self, in_channels, bias=False, eq_lr=False, spectral_normalization=False):
        return nn.Sequential(
            bb.Conv2d(
                in_channels=in_channels,
                out_channels=self.hparams.image_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
                eq_lr=eq_lr,
                spectral_normalization=False
            )
        )

    def forward(self, x, y):
        outputs = []
        x = x.view(x.size(0), -1, 1, 1)

        for block, to_rgb in zip(self.blocks, self.to_rgb_converts):
            x = block(x)
            outputs.append(torch.tanh(to_rgb(x)))

        return outputs
