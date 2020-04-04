import math

import pytorch_lightning as pl
import torch
import torch.nn as nn


class SimpleCombiner(nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class LinCatCombiner(nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()

        # TODO: to fix

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=self.hparams.image_channels,
            out_channels=self.hparams.image_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)

        return torch.cat([x1, x2], dim=1)


class CatLinCombiner(nn.Module):
    def __init__(self, hparams, in_channels):
        super().__init__()

        self.hparams = hparams
        self.in_channels = in_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels + self.hparams.image_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class SimpleCritic(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        if self.hparams.multi_scale_gradient:
            if self.hparams.multi_scale_gradient_combiner == "simple":
                additional_channels = self.hparams.image_channels
            elif self.hparams.multi_scale_gradient_combiner == "cat_lin":
                additional_channels = 0
            elif self.hparams.multi_scale_gradient_combiner == "lin_cat":
                additional_channels = self.hparams.image_channels
            else:
                raise ValueError()
        else:
            additional_channels = 0

        self.block1 = self.block_fn(self.hparams.image_channels, self.hparams.critic_filters)  # in: 64 x 64, out: 32 x 32
        self.block2 = self.block_fn(self.hparams.critic_filters + additional_channels, self.hparams.critic_filters * 2)  # in: 32 x 32, out: 16 x 16
        self.block3 = self.block_fn(self.hparams.critic_filters * 2 + additional_channels, self.hparams.critic_filters * 4)  # in: 16 x 16, out: 8 x 8
        self.block4 = self.block_fn(self.hparams.critic_filters * 4 + additional_channels, self.hparams.critic_filters * 8)  # in: 8 x 8, out: 4 x 4

        self.validator = nn.Conv2d(self.hparams.critic_filters * 8 + additional_channels, 1, 4, 1, 0, bias=False)

        self.combine1 = self.combine_fn(self.hparams.critic_filters)
        self.combine2 = self.combine_fn(self.hparams.critic_filters * 2)
        self.combine3 = self.combine_fn(self.hparams.critic_filters * 4)
        self.combine4 = self.combine_fn(self.hparams.critic_filters * 8)

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
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def combine_fn(self, in_channels):
        if self.hparams.multi_scale_gradient_combiner == "simple":
            return SimpleCombiner(self.hparams, in_channels)
        elif self.hparams.multi_scale_gradient_combiner == "lin_cat":
            return LinCatCombiner(self.hparams, in_channels)
        elif self.hparams.multi_scale_gradient_combiner == "cat_lin":
            return CatLinCombiner(self.hparams, in_channels)
        else:
            raise ValueError()

    # Dropout is just used for WGAN-CT
    def forward(self, x, y, dropout=0.0, intermediate_output=False, scaled_inputs=None):
        x_32x32 = self.block1(x)
        if self.hparams.multi_scale_gradient: x_32x32 = self.combine1(scaled_inputs[3], x_32x32)
        x_32x32 = torch.dropout(x_32x32, p=dropout, train=True)

        x_16x16 = self.block2(x_32x32)
        if self.hparams.multi_scale_gradient: x_16x16 = self.combine2(scaled_inputs[2], x_16x16)
        x_16x16 = torch.dropout(x_16x16, p=dropout, train=True)

        x_8x8 = self.block3(x_16x16)
        if self.hparams.multi_scale_gradient: x_8x8 = self.combine3(scaled_inputs[1], x_8x8)
        x_8x8 = torch.dropout(x_8x8, p=dropout, train=True)

        x_4x4 = self.block4(x_8x8)
        if self.hparams.multi_scale_gradient: x_4x4 = self.combine4(scaled_inputs[0], x_4x4)
        x_4x4 = torch.dropout(x_4x4, p=dropout, train=True)

        validity = self.validator(x_4x4)

        if intermediate_output:
            return validity, x.mean()
        else:
            return validity
