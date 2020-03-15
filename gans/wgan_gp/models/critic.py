import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.image_channels = self.hparams.image_channels
        self.image_size = self.hparams.image_size
        self.leaky_relu_slope = self.hparams.critic_leaky_relu_slope

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.image_channels, self.image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            # state size. (self.image_size) x 32 x 32
            nn.Conv2d(self.image_size, self.image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.image_size * 2),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*2) x 16 x 16
            nn.Conv2d(self.image_size * 2, self.image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.image_size * 4),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*4) x 8 x 8
            nn.Conv2d(self.image_size * 4, self.image_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.image_size * 8),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            # state size. (self.image_size*8) x 4 x 4
            nn.Conv2d(self.image_size * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        x = self.main(x)

        return x.squeeze()
