import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from gans.architectures.HDCGAN import DownsampleHDCGANBlock, LastHDCGANBlock
from gans.architectures.PROGAN import DownsampleProGANBlock, LastProGANBlock
from gans.init import snn_weight_init, he_weight_init


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

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=self.hparams.image_channels,
            out_channels=self.hparams.image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        if self.hparams.architecture == "progan":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif self.hparams.architecture == "hdcgan":
            self.activation = nn.SELU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.activation(x1)

        return torch.cat([x1, x2], dim=1)


class CatLinCombiner(nn.Module):
    def __init__(self, hparams, in_channels, bias=False):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels + self.hparams.image_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        if self.hparams.architecture == "progan":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif self.hparams.architecture == "hdcgan":
            self.activation = nn.SELU(inplace=True)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.activation(x)

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

        if self.hparams.exponential_filter_multipliers:
            self.filter_multipliers = [
                2 ** (x + 1)
                for x in range(1, int(math.log2(self.hparams.image_size)))
            ]
        else:
            self.filter_multipliers = [
                1
                for x in range(1, int(math.log2(self.hparams.image_size)))
            ]

            self.filter_multipliers[0] = 1

        self.blocks.append(
            self.block_fn(
                self.hparams.image_channels,
                self.filter_multipliers[0] * self.hparams.discriminator_filters,
                self.bias
            )
        )

        self.from_rgb_combiners.append(
            self.from_rgb_fn(
                self.filter_multipliers[0] * self.hparams.discriminator_filters,
                self.bias
            )
        )

        for pos, _ in enumerate(self.filter_multipliers[1:-1]):
            self.blocks.append(
                self.block_fn(
                    self.filter_multipliers[pos] * self.hparams.discriminator_filters + additional_channels,
                    self.filter_multipliers[pos + 1] * self.hparams.discriminator_filters,
                    self.bias
                )
            )

            self.from_rgb_combiners.append(
                self.from_rgb_fn(
                    self.filter_multipliers[pos + 1] * self.hparams.discriminator_filters,
                    self.bias
                )
            )

        # Validation part
        if self.hparams.architecture == "progan":
            self.blocks.append(
                LastProGANBlock(
                    in_channels=self.filter_multipliers[-2] * self.hparams.discriminator_filters,
                    out_channels=self.filter_multipliers[-1] * self.hparams.discriminator_filters,
                    additional_channels=additional_channels,
                    bias=self.bias
                )
            )
        elif self.hparams.architecture == "hdcgan":
            self.blocks.append(
                LastHDCGANBlock(
                    in_channels=self.filter_multipliers[-2] * self.hparams.discriminator_filters,
                    out_channels=self.filter_multipliers[-1] * self.hparams.discriminator_filters,
                    additional_channels=additional_channels,
                    bias=self.bias
                )
            )

        if self.hparams.weight_init == "he":
            self.apply(he_weight_init)
        elif self.hparams.weight_init == "snn":
            self.apply(snn_weight_init)

        if self.hparams.spectral_normalization:
            for block in self.blocks:
                block.conv1 = spectral_norm(block.conv1)
                block.conv2 = spectral_norm(block.conv2)

    def block_fn(self, in_channels, out_channels, bias=False):
        if self.hparams.architecture == "progan":
            return DownsampleProGANBlock(in_channels, out_channels, bias=bias)
        elif self.hparams.architecture == "hdcgan":
            return DownsampleHDCGANBlock(in_channels, out_channels, bias=bias)

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
        if isinstance(x, list):
            # msg enabled
            last_x_forward = None
            x = list(reversed(x))

            x_forward = self.blocks[0](x[0])

            for data, block, from_rgb in zip(x[1:], self.blocks[1:], self.from_rgb_combiners):
                last_x_forward = x_forward

                x_forward = from_rgb(data, x_forward)
                x_forward = torch.dropout(x_forward, p=dropout, train=True)
                x_forward = block(x_forward)

            if intermediate_output:
                return x_forward, last_x_forward.mean()
            else:
                return x_forward
        else:
            last_x_forward = None
            x_forward = x

            for block in self.blocks:
                last_x_forward = x
                x_forward = block(x_forward)

            if intermediate_output:
                return x_forward, last_x_forward.mean()
            else:
                return x_forward
