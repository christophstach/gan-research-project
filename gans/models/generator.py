import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from ..building_blocks import bb.SelfAttention2d, bb.PixelNorm, nn.Conv2d, bb.ConvTranspose2d

import gans.building_blocks as bb


class UpsampleSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.upsample(x)
        x = F.leaky_relu(self.conv(x), 0.2)

        return x


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.pixelNorm = bb.PixelNorm()

    def forward(self, x):
        x = self.upsample(x)
        x = self.pixelNorm(F.leaky_relu(self.conv_skip(x), 0.2))

        identity = x
        x = self.pixelNorm(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pixelNorm(F.leaky_relu(self.conv2(x), 0.2))

        return x + identity


class UpsampleSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.att = bb.SelfAttention2d(out_channels, bias=bias)

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
                bb.ConvTranspose2d(
                    self.hparams.noise_size,
                    self.hparams.generator_filters,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=self.bias
                ),
                nn.LeakyReLU(0.2, inplace=True)
                # bb.PixelNorm()
            )
        )

        for i in range(2, int(math.log2(self.hparams.image_size))):
            self.blocks.append(
                self.block_fn(
                    self.hparams.generator_filters // 2 ** (i - 2),
                    self.hparams.generator_filters // 2 ** (i - 1),
                    self.bias
                )
            )

        for i in range(1, int(math.log2(self.hparams.image_size))):
            self.to_rgb_converts.append(
                self.to_rgb_fn(
                    self.hparams.generator_filters // 2 ** (i - 1),
                    self.bias
                )
            )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if self.hparams.weight_init == "dcgan":
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        elif self.hparams.weight_init == "he":
            if isinstance(m, nn.Conv2d):
                # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                # https://arxiv.org/abs/1502.01852

                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def block_fn(self, in_channels, out_channels, bias=False):
        # return UpsampleSelfAttentionBlock(in_channels, out_channels, bias=bias)
        # return UpsampleResidualBlock(in_channels, out_channels, bias=bias)
        return UpsampleSimpleBlock(in_channels, out_channels, bias=bias)

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

        for block, to_rgb in zip(self.blocks, self.to_rgb_converts):
            x = block(x)
            outputs.append(torch.tanh(to_rgb(x)))

        return outputs
