import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from gans.architectures.HDCGAN import FirstHDCGANBlock, UpsampleHDCGANBlock
from gans.architectures.PROGAN import FirstProGANBlock, UpsampleProGANBlock
from gans.init import snn_weight_init, he_weight_init


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.bias = True

        self.blocks = nn.ModuleList()
        self.to_rgb_converts = nn.ModuleList()
        self.z_skip_connections = nn.ModuleList()

        if self.hparams.exponential_filter_multipliers:
            self.filter_multipliers = [
                2 ** (x + 1)
                for x in reversed(range(1, int(math.log2(self.hparams.image_size))))
            ]
        else:
            self.filter_multipliers = [
                1
                for x in range(1, int(math.log2(self.hparams.image_size)))
            ]

            self.filter_multipliers[-1] = 1

        if self.hparams.architecture == "progan":
            self.blocks.append(
                FirstProGANBlock(
                    noise_size=self.hparams.noise_size,
                    filters=self.filter_multipliers[0] * self.hparams.generator_filters,
                    bias=self.bias
                )
            )
        elif self.hparams.architecture == "hdcgan":
            self.blocks.append(
                FirstHDCGANBlock(
                    noise_size=self.hparams.noise_size,
                    filters=self.filter_multipliers[0] * self.hparams.generator_filters,
                    bias=self.bias
                )
            )

        self.to_rgb_converts.append(
            self.to_rgb_fn(
                self.filter_multipliers[0] * self.hparams.generator_filters,
                self.bias
            )
        )

        self.z_skip_connections.append(
            None
        )

        for pos, _ in enumerate(self.filter_multipliers[1:]):
            self.blocks.append(
                self.block_fn(
                    self.filter_multipliers[pos] * self.hparams.generator_filters,
                    self.filter_multipliers[pos + 1] * self.hparams.generator_filters,
                    self.bias
                )
            )

            self.to_rgb_converts.append(
                self.to_rgb_fn(
                    self.filter_multipliers[pos + 1] * self.hparams.generator_filters,
                    self.bias
                )
            )
          
            self.z_skip_connections.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hparams.noise_size,
                        self.hparams.generator_filters * self.filter_multipliers[pos + 1],
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=self.bias
                    ),
                    nn.UpsamplingNearest2d(scale_factor=2 ** (pos + 1))
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
            return UpsampleProGANBlock(in_channels, out_channels, bias=bias)
        elif self.hparams.architecture == "hdcgan":
            return UpsampleHDCGANBlock(in_channels, out_channels, bias=bias)

    def to_rgb_fn(self, in_channels, bias=False):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hparams.image_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias
            )
        )

    def forward(self, x, y):
        outputs = []
        x = x.view(x.size(0), -1, 1, 1)
        z = x

        for i, (block, to_rgb, z_skip) in enumerate(zip(self.blocks, self.to_rgb_converts, self.z_skip_connections)):
            x = block(x)

            if i > 0: x = x + z_skip(z)

            output = torch.tanh(to_rgb(x))
            outputs.append(output)

        return outputs
