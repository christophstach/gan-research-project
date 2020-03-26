import pytorch_lightning as pl
import torch.nn as nn


class SimpleGenerator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hparams.noise_size, self.hparams.generator_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters * 8),
            nn.ReLU(True),
            # state size. (self.hparams.generator_filters*8) x 4 x 4
            nn.ConvTranspose2d(self.hparams.generator_filters * 8, self.hparams.generator_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters * 4),
            nn.ReLU(True),
            # state size. (self.hparams.generator_filters*4) x 8 x 8
            nn.ConvTranspose2d(self.hparams.generator_filters * 4, self.hparams.generator_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters * 2),
            nn.ReLU(True),
            # state size. (self.hparams.generator_filters*2) x 16 x 16
            nn.ConvTranspose2d(self.hparams.generator_filters * 2, self.hparams.generator_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.generator_filters),
            nn.ReLU(True),
            # state size. (self.hparams.generator_filters) x 32 x 32
            nn.ConvTranspose2d(self.hparams.generator_filters, self.hparams.image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if True:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x, y):
        x = x.view(x.size(0), -1, 1, 1)
        data = self.main(x)

        return data
