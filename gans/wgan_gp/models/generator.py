import pytorch_lightning as pl
import torch.nn as nn


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hparams.noise_size, self.hparams.image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams.image_size * 8),
            nn.ReLU(True),
            # state size. (self.image_size*8) x 4 x 4
            nn.ConvTranspose2d(self.hparams.image_size * 8, self.hparams.image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.image_size * 4),
            nn.ReLU(True),
            # state size. (self.image_size*4) x 8 x 8
            nn.ConvTranspose2d(self.hparams.image_size * 4, self.hparams.image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.image_size * 2),
            nn.ReLU(True),
            # state size. (self.image_size*2) x 16 x 16
            nn.ConvTranspose2d(self.hparams.image_size * 2, self.hparams.image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.image_size),
            nn.ReLU(True),
            # state size. (self.image_size) x 32 x 32
            nn.ConvTranspose2d(self.hparams.image_size, self.hparams.image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x, y):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.main(x)

        return x
