import pytorch_lightning as pl
import torch
import torch.nn as nn


class Generator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hparams.noise_size + self.hparams.y_embedding_size, self.hparams.image_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hparams.image_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams.image_size * 4, self.hparams.image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.image_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams.image_size * 2, self.hparams.image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hparams.image_size),
            nn.ReLU(True),

            #            nn.ConvTranspose2d(self.hparams.image_size * 2, self.hparams.image_size, 4, 2, 1, bias=False),
            #            nn.BatchNorm2d(self.hparams.image_size),
            #            nn.ReLU(True),

            nn.ConvTranspose2d(self.hparams.image_size, self.hparams.image_channels, 4, 2, 1, bias=False),
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
