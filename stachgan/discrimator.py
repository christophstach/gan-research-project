import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_width = self.hparams.image_width
        self.image_height = self.hparams.image_height
        self.leaky_relu_slope = self.hparams.discriminator_leaky_relu_slope
        self.dropout_rate = self.hparams.discriminator_dropout_rate
        self.filters = self.hparams.discriminator_filters
        self.latent_dim = self.hparams.discriminator_latent_dim

        self.features = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.image_channels, self.filters, kernel_size=3),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.filters, self.filters * 2, kernel_size=3),
            nn.BatchNorm2d(self.filters * 2),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.filters * 2, self.filters * 4, kernel_size=3),
            nn.BatchNorm2d(self.filters * 4),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.image_width * self.image_height * self.filters * 4, self.latent_dim * 4),
            nn.BatchNorm1d(self.latent_dim * 4),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),
            nn.Dropout(p=self.dropout_rate, inplace=False),

            nn.Linear(self.latent_dim * 4, self.latent_dim * 2),
            nn.BatchNorm1d(self.latent_dim * 2),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),
            nn.Dropout(p=self.dropout_rate, inplace=False),

            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),

            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
