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
            nn.Conv2d(self.image_channels, self.filters, kernel_size=3, bias=False),
            nn.BatchNorm2d(self.filters),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.filters, self.filters * 2, kernel_size=3, bias=False),
            nn.BatchNorm2d(self.filters * 2),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope, inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(self.filters * 2, self.filters * 4, kernel_size=3, bias=False),
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

# From pytorch.org
class SimpleDiscriminator(nn.Module):
    def __init__(self, hparams):
        super(SimpleDiscriminator, self).__init__()
        self.hparams = hparams

        # Number of channels in the training images. For color images this is 3
        nc = self.hparams.image_channels
        # Size of feature maps in discriminator
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)