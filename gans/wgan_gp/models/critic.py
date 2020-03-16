import pytorch_lightning as pl
import torch.nn as nn


class Critic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.hparams.image_channels, self.hparams.image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.image_size) x 32 x 32
            nn.Conv2d(self.hparams.image_size, self.hparams.image_size * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.image_size * 2),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*2) x 16 x 16
            nn.Conv2d(self.hparams.image_size * 2, self.hparams.image_size * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.image_size * 4),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*4) x 8 x 8
            nn.Conv2d(self.hparams.image_size * 4, self.hparams.image_size * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.image_size * 8),
            nn.LeakyReLU(self.hparams.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*8) x 4 x 4
            nn.Conv2d(self.hparams.image_size * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        x = self.main(x)
        x = x.squeeze()

        return x
