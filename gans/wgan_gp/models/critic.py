import math

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.leaky_relu_slope = self.hparams.critic_leaky_relu_slope
        self.filters = self.hparams.critic_filters
        self.length = self.hparams.critic_length
        self.y_size = self.hparams.y_size
        self.y_embedding_size = self.hparams.y_embedding_size if self.y_size > 0 else 0

        self.main = nn.Sequential(
            self.first_block(),
            *[self.middle_block(block_idx) for block_idx in range(self.length)],
            self.last_block()
        )

        self.y_embedding = nn.Embedding(num_embeddings=self.y_size, embedding_dim=self.image_width * self.image_height)

    def first_block(self):
        return nn.Sequential(
            nn.Conv2d(
                self.image_channels + 1 if self.y_size > 0 else self.image_channels,
                self.filters,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True)
        )

    def middle_block(self, block_idx):
        return nn.Sequential(
            nn.Conv2d(
                int(self.filters * math.pow(2, block_idx)),
                int(self.filters * math.pow(2, block_idx + 1)),
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(int(self.filters * math.pow(2, block_idx + 1))),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True)
        )

    def last_block(self):
        return nn.Sequential(
            nn.Conv2d(
                int(self.filters * math.pow(2, self.length)),
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x, y):
        if self.y_size > 0:
            y = self.y_embedding(y)
            # reshape embedding  so it can be used as a image layer and fed into the classifier
            y = y.view(x.size(0), -1, x.size(2), x.size(3))
            data = torch.cat((x, y), dim=1)
        else:
            data = x

        data = self.main(data)

        return data
