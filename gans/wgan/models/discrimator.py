import math

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.leaky_relu_slope = self.hparams.discriminator_leaky_relu_slope
        self.filters = self.hparams.discriminator_filters
        self.length = self.hparams.discriminator_length

        self.main = nn.Sequential(
            self.first_block(),
            *[self.middle_block(block_idx) for block_idx in range(self.length)],
            self.last_block()
        )

    def first_block(self):
        return nn.Sequential(
            nn.Conv2d(
                self.image_channels,
                self.filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True)
        )

    def middle_block(self, block_idx):
        return nn.Sequential(
            nn.Conv2d(
                int(self.filters * math.pow(2, block_idx)),
                int(self.filters * math.pow(2, block_idx + 1)),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, block_idx + 1))),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True)
        )

    def last_block(self):
        return nn.Sequential(
            nn.Conv2d(
                int(self.filters * math.pow(2, self.length)),
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)

        return x
