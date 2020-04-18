from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from torch.nn.modules.utils import _pair


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", eq_lr=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if eq_lr:
            fan_in = prod(_pair(kernel_size)) * in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

    def forward(self, x):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(x, self._padding_repeated_twice, mode=self.padding_mode),
                self.weight * self.weight_scale,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups
            )
        return F.conv2d(
            x,
            self.weight * self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", eq_lr=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

        if eq_lr:
            fan_in = in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            x,
            self.weight * self.weight_scale,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation
        )


##########


class Conv2dOld(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", eq_lr=False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *_pair(kernel_size)),
            requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels),
                requires_grad=True
            )
        else:
            self.bias = None

        if eq_lr:
            fan_in = prod(_pair(kernel_size)) * in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.2, nonlinearity="leaky_relu")
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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


class ConvTranspose2dOld(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", eq_lr=False):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, *_pair(kernel_size)),
            requires_grad=True
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_channels),
                requires_grad=True
            )
        else:
            self.bias = None

        if eq_lr:
            fan_in = in_channels
            self.weight_scale = sqrt(2) / sqrt(fan_in)
        else:
            self.weight_scale = 1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.2, nonlinearity="leaky_relu")
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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
