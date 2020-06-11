import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from gans.architectures.HDCGAN import FirstHDCGANBlock, UpsampleHDCGANBlock
from gans.architectures.PROGAN import FirstProGANBlock, UpsampleProGANBlock
from gans.init import selu_weight_init, he_weight_init, orthogonal_weight_init

class ZSkipConnector(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x, z):
        height_multiplier = x.size(2) // z.size(2)
        width_multiplier = x.size(3) // z.size(3)

        z_repeated = z.repeat(1, 1, height_multiplier, width_multiplier)
        z_repeated = self.conv(z_repeated)

        return x + z_repeated


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.bias = True

        self.blocks = nn.ModuleList()
        self.to_rgb_converts = nn.ModuleList()
        self.z_skip_connections = nn.ModuleList()

        if self.hparams.exponential_filter_multipliers:
            self.filters = [
                2 ** (x + 1) * self.hparams.generator_filters
                for x in reversed(range(1, int(math.log2(self.hparams.image_size))))
            ]

            self.filters[0] = self.filters[1]
            self.filters[-1] = self.filters[-2]
        else:
            self.filters = [
                self.hparams.generator_filters
                for x in range(1, int(math.log2(self.hparams.image_size)))
            ]

        if self.hparams.architecture == "progan":
            self.blocks.append(
                FirstProGANBlock(
                    noise_size=self.hparams.noise_size,
                    filters=self.filters[0],
                    bias=self.bias
                )
            )
        elif self.hparams.architecture == "hdcgan":
            self.blocks.append(
                FirstHDCGANBlock(
                    noise_size=self.hparams.noise_size,
                    filters=self.filters[0],
                    bias=self.bias
                )
            )

        self.to_rgb_converts.append(
            self.to_rgb_fn(
                self.filters[0],
                self.bias
            )
        )

        self.z_skip_connections.append(
            None
        )

        for pos, _ in enumerate(self.filters[1:]):
            self.blocks.append(
                self.block_fn(
                    self.filters[pos],
                    self.filters[pos + 1],
                    self.bias
                )
            )

            self.to_rgb_converts.append(
                self.to_rgb_fn(
                    self.filters[pos + 1],
                    self.bias
                )
            )
          
            self.z_skip_connections.append(
                self.z_skip_connection_fn(
                    self.hparams.noise_size // 64,
                    self.filters[pos + 1],
                    self.bias
                )
            )
       
        if self.hparams.weight_init == "he":
            self.apply(he_weight_init)
        elif self.hparams.weight_init == "selu":
            self.apply(selu_weight_init)
        elif self.hparams.weight_init:
            self.apply(orthogonal_weight_init)

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

    def z_skip_connection_fn(self, in_channels, out_channels, bias=False):
        return ZSkipConnector(in_channels, out_channels, bias)

    def forward(self, x, y):
        outputs = []
        x = x.view(x.size(0), -1, 1, 1)
        # z = x.view(x.size(0), -1, 8, 8)

        for block, to_rgb, z_skip in zip(self.blocks, self.to_rgb_converts, self.z_skip_connections):
            x = block(x)
            
            # if x.size(2) >= 8 and x.size(3) >= 8:
            #    x = z_skip(x, z)

            output = torch.tanh(to_rgb(x))
            outputs.append(output)

        return outputs
