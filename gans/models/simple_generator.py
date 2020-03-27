import math

import pytorch_lightning as pl
import torch.nn as nn


class SimpleGenerator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hparams.noise_size, self.hparams.generator_filters * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.generator_filters*4) x 4 x 4
            nn.ConvTranspose2d(self.hparams.generator_filters * 4, self.hparams.generator_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.generator_filters*2) x 8 x 8
            nn.ConvTranspose2d(self.hparams.generator_filters * 2, self.hparams.generator_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.generator_filters) x 32 x 32
            nn.ConvTranspose2d(self.hparams.generator_filters, self.hparams.image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (self.hparams.image_channels) x 32 x 32
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

    def forward(self, x, y):
        x = x.view(x.size(0), -1, 1, 1)
        data = self.main(x)

        return data
