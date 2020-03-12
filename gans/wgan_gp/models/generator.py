import torch
import torch.nn as nn

from ...building_blocks import Conv2dPixelShuffle


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.noise_size = self.hparams.noise_size
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.filters = self.hparams.generator_filters
        self.y_size = self.hparams.y_size
        self.y_embedding_size = self.hparams.y_embedding_size if self.y_size > 0 else 0

        self.y_embedding = nn.Embedding(num_embeddings=self.y_size, embedding_dim=self.y_embedding_size)

        self.main = nn.Sequential(
            Conv2dPixelShuffle(self.noise_size + self.y_embedding_size, out_channels=self.filters * 4, kernel_size=1, upscale_factor=4),
            nn.PReLU(self.filters * 4),
            Conv2dPixelShuffle(self.filters * 4, out_channels=self.filters * 2, kernel_size=3, upscale_factor=2),
            nn.PReLU(self.filters * 2),
            Conv2dPixelShuffle(self.filters * 2, out_channels=self.filters, kernel_size=3, upscale_factor=2),
            nn.PReLU(self.filters),
            Conv2dPixelShuffle(self.filters, out_channels=self.image_channels, kernel_size=5, upscale_factor=2),
            nn.Tanh()
        )

    def forward(self, x, y):
        # For Conditional GAN add additional data to the original input data
        if self.y_size > 0:
            y = self.y_embedding(y)
            # reshape embedding  so it can be added on top of the noise vector
            y = y.view(x.size(0), -1, x.size(2), x.size(3))
            data = torch.cat((x, y), dim=1)
        else:
            data = x

        data = self.main(data)

        return data
