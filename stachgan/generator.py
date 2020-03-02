import math

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.noise = self.hparams.noise
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

    def first_block(self):
        return nn.Sequential(
            nn.ConvTranspose2d(
                self.noise,
                int(self.filters * math.pow(2, self.length)),
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length))),
            nn.ReLU(inplace=True)
        )

    def middle_block(self, block_idx):
        return nn.Sequential(
            nn.ConvTranspose2d(
                int(self.filters * math.pow(2, self.length - block_idx)),
                int(self.filters * math.pow(2, self.length - block_idx - 1)),
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length - block_idx - 1))),
            nn.ReLU(inplace=True)
        )

    def last_block(self):
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

    def forward(self, x):
        x = self.main(x)

        return x

# From pytorch.org
class SimpleGenerator(nn.Module):
    def __init__(self, hparams):
        super(SimpleGenerator, self).__init__()

        self.hparams = hparams
        # Number of channels in the training images. For color images this is 3
        nc = self.hparams.image_channels
        # Size of z latent vector (i.e. size of generator input)
        nz = self.hparams.noise
        # Size of feature maps in generator
        ngf = 64

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)