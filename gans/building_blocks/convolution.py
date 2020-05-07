from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.utils import spectral_norm


class SubPixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, upscale_factor=2, bias=True, padding_mode="zeros", eq_lr=False, spectral_normalization=True):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels * 2 ** upscale_factor,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
            eq_lr=eq_lr,
            spectral_normalization=spectral_normalization
        )

        self.pixelShuffle = nn.PixelShuffle(self.upscale_factor)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, _Conv2d):
            # ICNR: https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
            new_shape = [int(m.weight.shape[0] / (self.upscale_factor ** 2))] + list(m.weight.shape[1:])
            sub_kernel = torch.zeros(new_shape)
            torch.nn.init.kaiming_normal_(sub_kernel)
            sub_kernel = sub_kernel.transpose(0, 1)

            sub_kernel = sub_kernel.contiguous().view(sub_kernel.shape[0], sub_kernel.shape[1], -1)
            kernel = sub_kernel.repeat(1, 1, self.upscale_factor ** 2)

            transposed_shape = [m.weight.shape[1]] + [m.weight.shape[0]] + list(m.weight.shape[2:])
            kernel = kernel.contiguous().view(transposed_shape)

            kernel = kernel.transpose(0, 1)
            m.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelShuffle(x)

        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", eq_lr=False, spectral_normalization=False):
        super().__init__()

        if spectral_normalization:
            self.conv = spectral_norm(_Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, eq_lr))
        else:
            self.conv = _Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, eq_lr)

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", eq_lr=False, spectral_normalization=True):
        super().__init__()

        if spectral_normalization:
            self.convTranspose = spectral_norm(_ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, eq_lr))
        else:
            self.convTranspose = _ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, eq_lr)

    def forward(self, x):
        return self.convTranspose(x)


class _Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", eq_lr=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        if eq_lr:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            self.weight_scale = sqrt(2.0 / fan_in)
        else:
            self.weight_scale = 1.0

    def forward(self, x):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._padding_repeated_twice, mode=self.padding_mode),
                self.weight * self.weight_scale,
                self.bias * self.weight_scale,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups
            )
        return F.conv2d(
            x,
            self.weight * self.weight_scale,
            self.bias * self.weight_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class _ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode="zeros", eq_lr=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

        if eq_lr:
            fan_in = in_channels
            self.weight_scale = sqrt(2.0 / fan_in)
        else:
            self.weight_scale = 1.0

    def forward(self, x, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose2d")

        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            x,
            self.weight * self.weight_scale,
            self.bias * self.weight_scale,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation
        )
