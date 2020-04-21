import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gans.building_blocks as bb


class SimpleCombiner(nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class LinCatCombiner(nn.Module):
    def __init__(self, hparams, in_channels, bias=False):
        super().__init__()

        # TODO: to fix

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = bb.Conv2d(
            in_channels=self.hparams.image_channels,
            out_channels=self.hparams.image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)

        return torch.cat([x1, x2], dim=1)


class CatLinCombiner(nn.Module):
    def __init__(self, hparams, in_channels, bias=False):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = bb.Conv2d(
            in_channels=in_channels + self.hparams.image_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class DownsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv1 = bb.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = bb.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.downsample = bb.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=bias
        )

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.downsample(x + identity)

        return x


class DownsampleSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.downsample = bb.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=bias
        )

    def forward(self, x):
        x = F.leaky_relu(self.downsample(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.bias = True

        if self.hparams.multi_scale_gradient:
            if self.hparams.multi_scale_gradient_combiner == "simple":
                additional_channels = self.hparams.image_channels
            elif self.hparams.multi_scale_gradient_combiner == "cat_lin":
                additional_channels = 0
            elif self.hparams.multi_scale_gradient_combiner == "lin_cat":
                additional_channels = self.hparams.image_channels
            else:
                raise ValueError()
        else:
            additional_channels = 0

        self.blocks = nn.ModuleList()
        self.from_rgb_combiners = nn.ModuleList()

        self.blocks.append(
            self.block_fn(
                self.hparams.image_channels,
                self.hparams.discriminator_filters // 2 ** (int(math.log2(self.hparams.image_size)) - 3),
                self.bias
            )
        )

        # print(self.hparams.discriminator_filters // 2 ** (int(math.log2(self.hparams.image_size)) - 3))

        for i in range(2, int(math.log2(self.hparams.image_size)) - 1):
            o = int(math.log2(self.hparams.image_size)) - i

            self.blocks.append(
                self.block_fn(
                    self.hparams.discriminator_filters // 2 ** (o - 1) + additional_channels,
                    self.hparams.discriminator_filters // 2 ** (o - 2),
                    self.bias
                )
            )

            # print(
            #    self.hparams.discriminator_filters // 2 ** (o - 1),
            #    self.hparams.discriminator_filters // 2 ** (o - 2)
            # )

        for i in range(1, int(math.log2(self.hparams.image_size)) - 1):
            o = int(math.log2(self.hparams.image_size)) - i

            self.from_rgb_combiners.append(
                self.from_rgb_fn(
                    self.hparams.discriminator_filters // 2 ** (o - 2),
                    self.bias
                )
            )

        self.validator = nn.Sequential(
            bb.MinibatchStdDev(),
            bb.Conv2d(
                self.hparams.discriminator_filters + additional_channels + 1,
                self.hparams.discriminator_filters + additional_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias
            ),
            nn.LeakyReLU(0.2, inplace=True),
            bb.Conv2d(
                self.hparams.discriminator_filters + additional_channels,
                self.hparams.discriminator_filters + additional_channels,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=self.bias
            ),
            nn.LeakyReLU(0.2, inplace=True),
            bb.Conv2d(
                self.hparams.discriminator_filters + additional_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=self.bias
            )
        )

    def block_fn(self, in_channels, out_channels, bias=False):
        # return DownsampleResidualBlock(in_channels, out_channels, bias=bias)
        return DownsampleSimpleBlock(in_channels, out_channels, bias=bias)

    def from_rgb_fn(self, in_channels, bias=False):
        if self.hparams.multi_scale_gradient_combiner == "simple":
            return SimpleCombiner(self.hparams, in_channels)
        elif self.hparams.multi_scale_gradient_combiner == "lin_cat":
            return LinCatCombiner(self.hparams, in_channels, bias=bias)
        elif self.hparams.multi_scale_gradient_combiner == "cat_lin":
            return CatLinCombiner(self.hparams, in_channels, bias=bias)
        else:
            raise ValueError()

    # Dropout is just used for WGAN-CT
    def forward(self, x, y, dropout=0.0, intermediate_output=False):
        x_hats = None

        for i in range(1, int(math.log2(self.hparams.image_size)) - 1):
            if x_hats is None:
                if isinstance(x, list):
                    x_hats = [self.blocks[i - 1](x[-1])]
                else:
                    x_hats = [self.blocks[i - 1](x)]
            else:
                x_hats.append(self.blocks[i - 1](x_hats[-1]))

            if isinstance(x, list): x_hats[i - 1] = self.from_rgb_combiners[i - 1](x[len(x) - i - 1], x_hats[i - 1])
            x_hats[i - 1] = torch.dropout(x_hats[i - 1], p=dropout, train=True)

        validity = self.validator(x_hats[-1])

        if intermediate_output:
            return validity, x.mean()
        else:
            return validity
