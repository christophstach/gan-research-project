import math

import attn_gan_pytorch.CustomLayers as attn
import torch.nn as nn
import torch.nn.functional as F


class UpsampleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.upsample(x)

        identity = x
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.conv2(x)

        return x + identity


class UpsampleFullAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.attn1 = attn.FullAttention(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, use_batch_norm=False, use_spectral_norm=False)

    def forward(self, x):
        x = self.upsample(x)
        x, _ = self.attn1(x)

        return x


class Generator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.blocks = nn.ModuleList()
        self.to_rgb_converts = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(
                    self.hparams.noise_size,
                    self.hparams.generator_filters,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        for _ in range(2, int(math.log2(self.hparams.image_size)) - 1):
            self.blocks.append(self.block_fn(self.hparams.generator_filters, self.hparams.generator_filters))

        for _ in range(1, int(math.log2(self.hparams.image_size)) - 1):
            self.to_rgb_converts.append(self.rgb_fn(self.hparams.generator_filters))

        self.output = nn.Sequential(
            nn.ConvTranspose2d(self.hparams.generator_filters, self.hparams.image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if self.hparams.weight_init == "dcgan":
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        elif self.hparams.weight_init == "he":
            if isinstance(m, nn.Conv2d):
                # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
                # https://arxiv.org/abs/1502.01852

                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def block_fn(self, in_channels, out_channels):
        return UpsampleFullAttentionBlock(in_channels, out_channels)
        # return UpsampleResidualBlock(in_channels, out_channels)

    def rgb_fn(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.hparams.image_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x, y):
        x = x.view(x.size(0), -1, 1, 1)

        x_hats = None
        for i in range(1, int(math.log2(self.hparams.image_size)) - 1):
            if x_hats is None:
                x_hats = [self.blocks[i - 1](x)]
            else:
                x_hats.append(self.blocks[i - 1](x_hats[-1]))

        x = self.output(x_hats[-1])

        if self.hparams.multi_scale_gradient:
            return x, [
                self.to_rgb_converts[i](x_hat)
                for i, x_hat in enumerate(x_hats)
            ]
        else:
            return x
