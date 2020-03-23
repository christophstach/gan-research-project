import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..building_blocks import UpsampleInterpolateConv2d, UpsampleFractionalConv2d


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            UpsampleFractionalConv2d(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, kernel_size=4, stride=1),
            UpsampleInterpolateConv2d(self.hparams.generator_filters * 4, self.hparams.generator_filters * 2),
            UpsampleInterpolateConv2d(self.hparams.generator_filters * 2, self.hparams.generator_filters)
        )

        self.y_embedding = nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.y_embedding_size)
        self.out = nn.Sequential(
            UpsampleInterpolateConv2d(self.hparams.generator_filters, self.hparams.image_channels),
            nn.Tanh()
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, y):
        y = self.y_embedding(y)
        x = torch.cat([x, y], dim=1)

        # Comment lines above to disable conditional gans

        x = x.view(x.size(0), -1, 1, 1)

        x = self.main(x)

        x = self.out(x)

        return x