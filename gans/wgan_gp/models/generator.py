import pytorch_lightning as pl
import torch
import torch.nn as nn

from ...building_blocks import Conv2dPixelShuffle


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2dPixelShuffle(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, kernel_size=4, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(self.hparams.image_size * 4),
            nn.PReLU(),

            # Conv2dPixelShuffle(self.hparams.image_size * 4, self.hparams.image_size * 2, kernel_size=3, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size * 4, self.hparams.image_size * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.hparams.image_size * 2),
            nn.PReLU(),

            # Conv2dPixelShuffle(self.hparams.image_size * 2, self.hparams.image_size, kernel_size=3, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size * 2, self.hparams.image_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.hparams.image_size),
            nn.PReLU(),

            # Conv2dPixelShuffle(self.hparams.image_size, self.hparams.image_channels, kernel_size=3, upscale_factor=2),
            nn.ConvTranspose2d(self.hparams.image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.y_embedding = nn.Sequential(
            nn.Embedding(num_embeddings=self.hparams.y_size, embedding_dim=self.hparams.y_embedding_size)
        )

    def forward(self, x, y):
        y = self.y_embedding(y)
        x = torch.cat([x, y], dim=1)
        # Comment lines above to disable conditional gan

        x = x.view(x.size(0), -1, 1, 1)
        x = self.main(x)

        return x
