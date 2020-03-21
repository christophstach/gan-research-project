import pytorch_lightning as pl
import torch
import torch.nn as nn

from ...building_blocks import Conv2dPixelShuffle
import math


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2dPixelShuffle(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, kernel_size=4, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, kernel_size=4, stride=1, padding=0, bias=False),
            # nn.InstanceNorm2d(self.hparams.image_size * 4),
            nn.BatchNorm2d(self.hparams.image_size * 4),
            # nn.PReLU(),
            # nn.PReLU(),
            nn.ReLU6(inplace=True),

            # Conv2dPixelShuffle(self.hparams.image_size * 4, self.hparams.image_size * 2, kernel_size=5, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size * 4, self.hparams.image_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(self.hparams.image_size * 2),
            nn.BatchNorm2d(self.hparams.image_size * 2),
            # nn.PReLU(),
            # nn.PReLU(),
            nn.ReLU6(inplace=True),

            # Conv2dPixelShuffle(self.hparams.image_size * 2, self.hparams.image_size, kernel_size=5, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size * 2, self.hparams.image_size, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(self.hparams.image_size),
            nn.BatchNorm2d(self.hparams.image_size),
            # nn.PReLU(),
            # nn.PReLU(),
            nn.ReLU6(inplace=True),

            # Conv2dPixelShuffle(self.hparams.image_size, self.hparams.image_channels, kernel_size=5, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size, self.hparams.image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        self.y_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.y_embedding_size)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # https://arxiv.org/abs/1502.01852

            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x, y):
        y = self.y_embedding(y)
        x = torch.cat([x, y], dim=1)
        # Comment lines above to disable conditional gan

        x = x.view(x.size(0), -1, 1, 1)
        x = self.main(x)

        return x
