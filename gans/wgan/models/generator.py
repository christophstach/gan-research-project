import math

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.noise_size = self.hparams.noise_size
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.filters = self.hparams.generator_filters
        self.length = self.hparams.generator_length

        self.main = nn.Sequential(
            self.first_block(),
            *[self.middle_block(block_idx) for block_idx in range(self.length)],
            self.last_block()
        )

    def first_block_old(self):
        return nn.Sequential(
            nn.ConvTranspose2d(
                self.noise_size,
                int(self.filters * math.pow(2, self.length)),
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length))),
            nn.ReLU(inplace=True)
        )

    def first_block(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                self.noise_size,
                int(self.filters * math.pow(2, self.length)),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length))),
            nn.ReLU(inplace=True)
        )

    def middle_block_old(self, block_idx):
        return nn.Sequential(
            nn.ConvTranspose2d(
                int(self.filters * math.pow(2, self.length - block_idx)),
                int(self.filters * math.pow(2, self.length - block_idx - 1)),
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length - block_idx - 1))),
            nn.ReLU(inplace=True)
        )

    def middle_block(self, block_idx):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                int(self.filters * math.pow(2, self.length - block_idx)),
                int(self.filters * math.pow(2, self.length - block_idx - 1)),
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length - block_idx - 1))),
            nn.ReLU(inplace=True)
        )

    def last_block_old(self):
        return nn.Sequential(
            nn.ConvTranspose2d(
                self.filters,
                self.image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Sigmoid(),
        )

    def last_block(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                self.filters,
                self.image_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.main(x)

        return x
