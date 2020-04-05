import math

import pytorch_lightning as pl
import torch.nn as nn


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.block1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                self.hparams.noise_size,
                self.hparams.generator_filters * 16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block2 = self.block_fn(self.hparams.generator_filters * 16, self.hparams.generator_filters * 8)  # in: 4 x 4, out: 8 x 8
        self.block3 = self.block_fn(self.hparams.generator_filters * 8, self.hparams.generator_filters * 4)  # in: 8 x 8, out: 16 x 16
        self.block4 = self.block_fn(self.hparams.generator_filters * 4, self.hparams.generator_filters * 2)  # in: 16 x 16, out: 32 x 32
        self.block5 = self.block_fn(self.hparams.generator_filters * 2, self.hparams.generator_filters)  # in: 32 x 32, out: 64 x 64

        self.output = nn.Sequential(
            nn.ConvTranspose2d(self.hparams.generator_filters, self.hparams.image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.rgb_4x4 = self.rgb_fn(self.hparams.generator_filters * 8)
        self.rgb_8x8 = self.rgb_fn(self.hparams.generator_filters * 4)
        self.rgb_16x16 = self.rgb_fn(self.hparams.generator_filters * 2)
        self.rgb_32x32 = self.rgb_fn(self.hparams.generator_filters)
        self.rgb_64x64 = self.rgb_fn(self.hparams.generator_filters)

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

    def block_fn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def rgb_fn(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hparams.image_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x, y):
        x = x.view(x.size(0), -1, 1, 1)

        x_4x4 = self.block1(x)
        x_8x8 = self.block2(x_4x4)
        x_16x16 = self.block3(x_8x8)
        x_32x32 = self.block4(x_16x16)
        x_64x64 = self.block5(x_32x32)

        x = self.output(x_64x64)

        if self.hparams.multi_scale_gradient:
            return x, [
                self.rgb_4x4(x_4x4),
                self.rgb_8x8(x_8x8),
                self.rgb_16x16(x_16x16),
                self.rgb_32x32(x_32x32),
                self.rgb_64x64(x_64x64)
            ]
        else:
            return x
