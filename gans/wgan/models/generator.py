import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.noise_size = self.hparams.noise_size
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.filters = self.hparams.generator_filters
        self.length = self.hparams.generator_length
        self.y_size = self.hparams.y_size
        self.y_embedding_size = self.hparams.y_embedding_size if self.y_size > 0 else 0

        self.y_embedding = nn.Embedding(num_embeddings=self.y_size, embedding_dim=self.y_embedding_size)

        self.main = nn.Sequential(
            self.first_block(),
            *[self.middle_block(block_idx) for block_idx in range(self.length)],
            self.last_block()
        )

    def first_block(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(
                self.noise_size + self.y_embedding_size,
                int(self.filters * math.pow(2, self.length)),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length))),
            nn.ReLU(inplace=True)
        )

    def middle_block(self, block_idx):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                int(self.filters * math.pow(2, self.length - block_idx)),
                int(self.filters * math.pow(2, self.length - block_idx - 1)),
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, self.length - block_idx - 1))),
            nn.ReLU(inplace=True)
        )

    def last_block(self):
        return nn.Sequential(
            nn.Upsample(size=(self.image_width, self.image_height)),
            nn.ConvTranspose2d(
                self.filters,
                self.image_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x, y):
        # For Conditional GAN add additional data to the original input data
        if self.y_size > 0:
            y = self.y_embedding(y)
            y = y.view(y.shape[0], y.shape[1], 1, 1)
            data = torch.cat((x, y), dim=1)
        else:
            data = x

        data = self.main(data)

        return data
