import math

import pytorch_lightning as pl
import torch
import torch.nn as nn


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SimpleCritic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.block1 = block(self.hparams.image_channels, self.hparams.critic_filters)  # in: 64 x 64, out: 32 x 32
        self.block2 = block(self.hparams.critic_filters, self.hparams.critic_filters * 2)  # in: 32 x 32, out: 16 x 16
        self.block3 = block(self.hparams.critic_filters * 2, self.hparams.critic_filters * 4)  # in: 16 x 16, out: 8 x 8
        self.block4 = block(self.hparams.critic_filters * 4, self.hparams.critic_filters * 8)  # in: 8 x 8, out: 4 x 4

        self.validator = nn.Conv2d(self.hparams.critic_filters * 8, 1, 4, 1, 0, bias=False)

        self.hparams = hparams

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

    def forward(self, x, y, dropout=0.0, intermediate_output=False):
        print(x.size())

        x = self.block1(x)
        x = torch.dropout(x, p=dropout, train=True)
        x = self.block2(x)
        x = torch.dropout(x, p=dropout, train=True)
        x = self.block3(x)
        x = torch.dropout(x, p=dropout, train=True)
        x = self.block4(x)
        x = torch.dropout(x, p=dropout, train=True)

        validity = self.validator(x)

        if intermediate_output:
            return validity, x.mean()
        else:
            return validity
