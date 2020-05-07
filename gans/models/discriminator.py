import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gans.building_blocks as bb
from gans.archictures.PROGAN import DownsampleProGANBlock, LastProGANBlock
from gans.init import snn_weight_init, he_weight_init


class SimpleCombiner(nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class LinCatCombiner(nn.Module):
    def __init__(self, hparams, in_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = bb.Conv2d(
            in_channels=self.hparams.image_channels,
            out_channels=self.hparams.image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )

    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.conv(x1), 0.2)

        return torch.cat([x1, x2], dim=1)


class CatLinCombiner(nn.Module):
    def __init__(self, hparams, in_channels, bias=False, eq_lr=False, spectral_normalization=False):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = bb.Conv2d(
            in_channels=in_channels + self.hparams.image_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)

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
        self.filter_multipliers = [
            2 ** (x + 1)
            for x in range(1, int(math.log2(self.hparams.image_size)))
        ]

        self.blocks.append(
            self.block_fn(
                self.hparams.image_channels,
                self.filter_multipliers[0] * self.hparams.discriminator_filters,
                self.bias,
                self.hparams.equalized_learning_rate,
                self.hparams.spectral_normalization
            )
        )

        self.from_rgb_combiners.append(
            self.from_rgb_fn(
                2 * self.filter_multipliers[0] * self.hparams.discriminator_filters,
                self.bias,
                self.hparams.equalized_learning_rate,
                self.hparams.spectral_normalization
            )
        )

        for i in self.filter_multipliers[1:-1]:
            self.blocks.append(
                self.block_fn(
                    i // 2 * self.hparams.discriminator_filters + additional_channels,
                    i * self.hparams.discriminator_filters,
                    self.bias,
                    self.hparams.equalized_learning_rate,
                    self.hparams.spectral_normalization
                )
            )

            self.from_rgb_combiners.append(
                self.from_rgb_fn(
                    2 * i * self.hparams.discriminator_filters,
                    self.bias,
                    self.hparams.equalized_learning_rate,
                    self.hparams.spectral_normalization
                )
            )

        # Validation part
        self.blocks.append(
            LastProGANBlock(
                filters=self.filter_multipliers[-1] * self.hparams.discriminator_filters,
                additional_channels=additional_channels,
                bias=self.bias,
                eq_lr=self.hparams.equalized_learning_rate,
                spectral_normalization=self.hparams.spectral_normalization
            )
        )

        if self.hparams.weight_init == "he":
            self.apply(he_weight_init)
        elif self.hparams.weight_init == "snn":
            self.apply(snn_weight_init)

    def block_fn(self, in_channels, out_channels, bias=False, eq_lr=False, spectral_normalization=False):
        return DownsampleProGANBlock(in_channels, out_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)

    def from_rgb_fn(self, in_channels, bias=False, eq_lr=False, spectral_normalization=False):
        if self.hparams.multi_scale_gradient_combiner == "simple":
            return SimpleCombiner(self.hparams, in_channels)
        elif self.hparams.multi_scale_gradient_combiner == "lin_cat":
            return LinCatCombiner(self.hparams, in_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
        elif self.hparams.multi_scale_gradient_combiner == "cat_lin":
            return CatLinCombiner(self.hparams, in_channels, bias=bias, eq_lr=eq_lr, spectral_normalization=spectral_normalization)
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
