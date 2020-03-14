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
        self.y_size = self.hparams.y_size
        self.y_embedding_size = self.hparams.y_embedding_size if self.y_size > 0 else 0

        self.main = nn.Sequential(
            nn.Conv2d(self.image_channels, self.filters, kernel_size=5, stride=2, padding=2, padding_mode="zero"),
            nn.PReLU(self.filters),
            nn.Conv2d(self.filters, self.filters * 2, kernel_size=5, stride=2, padding=2, padding_mode="zero"),
            nn.BatchNorm2d(self.filters * 2),
            nn.PReLU(self.filters * 2),
            nn.Conv2d(self.filters * 2, self.filters * 4, kernel_size=5, stride=2, padding=2, padding_mode="zero"),
            nn.BatchNorm2d(self.filters * 4),
            nn.PReLU(self.filters * 4),
            nn.Conv2d(self.filters * 4, self.filters * 8, kernel_size=5, stride=2, padding=2, padding_mode="zero"),
            nn.BatchNorm2d(self.filters * 8),
            nn.PReLU(self.filters * 8)
        )

        self.y_embedding = nn.Embedding(num_embeddings=self.y_size, embedding_dim=self.y_embedding_size)
        self.validation = nn.Sequential(
            # nn.Conv2d(self.filters * 8, 1, kernel_size=4, stride=1, padding=0)
            nn.Linear(self.filters * 8 * (int(self.image_width / 16) * int(self.image_height / 16)) + self.y_embedding_size, 1)
        )

    def forward(self, x, y):
        x = self.main(x)
        x = x.view(x.size(0), -1)

        if self.y_size > 0:
            y = self.y_embedding(y)
            # reshape embedding  so it can be used as a image layer and fed into the classifier
            data = torch.cat((x, y), dim=1)
        else:
            data = x

        data = self.validation(data)

        return data
