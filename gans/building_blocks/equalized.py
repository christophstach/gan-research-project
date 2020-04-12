import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


# TODO: This needs rework

# ==========================================================
# equalized learning rate blocks:
# extending Conv2d and ConvTranspose2d layers for equalized learning rate logic
# ==========================================================
class EqualizedLearningRateConv2d(nn.Module):
    """ conv2d with the concept of EqualizedLearningRate learning rate
        Args:
            :param in_channels: input channels
            :param out_channels:  output channels
            :param kernel_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param padding: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """ constructor for the class """

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(
            nn.init.normal_(
                torch.empty(out_channels, in_channels, *_pair(kernel_size))
            ),
            requires_grad=True
        )

        self.use_bias = bias
        self.stride = stride
        self.padding = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        fan_in = math.prod(_pair(kernel_size)) * in_channels  # value of fan_in
        self.scale = math.sqrt(2) / math.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """

        return F.conv2d(
            input=x,
            weight=self.weight * self.scale,  # scale the weight on runtime
            bias=self.bias if self.use_bias else None,
            stride=self.stride,
            padding=self.padding
        )

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedLearningRateConvTranspose2d(nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param in_channels: input channels
            :param out_channels: output channels
            :param kernel_size: kernel size
            :param stride: stride for convolution transpose
            :param padding: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(in_channels, out_channels, *_pair(kernel_size))
            ),
            requires_grad=True
        )

        self.use_bias = bias
        self.stride = stride
        self.padding = padding

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        fan_in = in_channels  # value of fan_in for deconv
        self.scale = math.sqrt(2) / math.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """

        return F.conv_transpose2d(
            input=x,
            weight=self.weight * self.scale,  # scale the weight on runtime
            bias=self.bias if self.use_bias else None,
            stride=self.stride,
            padding=self.padding
        )

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedLearningRateLinear(nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        Linear layer modified for equalized learning rate
        """

        super().__init__()

        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(out_channels, in_channels)
        ), requires_grad=True)

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        fan_in = in_channels
        self.scale = math.sqrt(2) / math.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        return F.linear(
            x,
            self.weight * self.scale,
            self.bias if self.use_bias else None
        )
