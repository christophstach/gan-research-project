import pytorch_lightning as pl
import torch.nn as nn


class SimpleCritic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.hparams.image_channels, self.hparams.critic_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters) x 32 x 32
            nn.Conv2d(self.hparams.critic_filters, self.hparams.critic_filters * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.critic_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*2) x 16 x 16
            nn.Conv2d(self.hparams.critic_filters * 2, self.hparams.critic_filters * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.critic_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*4) x 8 x 8
            nn.Conv2d(self.hparams.critic_filters * 4, self.hparams.critic_filters * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.hparams.critic_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.hparams.critic_filters*8) x 4 x 4
            nn.Conv2d(self.hparams.critic_filters * 8, 1, 4, 1, 0, bias=False),
        )

        self.hparams = hparams

    def init_weights(self, m):
        if True:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x, y):
        validity = self.main(x)

        return validity
