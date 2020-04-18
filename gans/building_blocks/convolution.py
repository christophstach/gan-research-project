from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from torch.nn.modules.utils import _pair


# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", eq_lr=False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            nn.init.normal_(
                torch.empty(out_channels, in_channels, *_pair(kernel_size))
            ),
            requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels),
                requires_grad=True
            )
        else:
            self.bias = None

        if eq_lr:
            fan_in = prod(_pair(kernel_size)) * in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", eq_lr=False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(in_channels, out_channels, *_pair(kernel_size))
            ),
            requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels),
                requires_grad=True
            )
        else:
            self.bias = None

        if eq_lr:
            fan_in = in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

    def forward(self, x):
        return F.conv_transpose2d(
            x,
            self.weight * self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation
        )
