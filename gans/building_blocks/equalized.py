import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: This needs rework

# ==========================================================
# Equalized learning rate blocks:
# extending Conv2D and Deconv2D layers for equalized learning rate logic
# ==========================================================
class EqualizedConv2d(nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ), requires_grad=True)

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

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
            padding=self.pad
        )

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedConvTranspose2d(nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(torch.nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ), requires_grad=True)

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = sqrt(2) / sqrt(fan_in)

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
            padding=self.pad
        )

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class EqualizedLinear(nn.Module):
    """ Linear layer using equalized learning rate
        Args:
            :param c_in: number of input channels
            :param c_out: number of output channels
            :param bias: whether to use bias with the linear layer
    """

    def __init__(self, c_in, c_out, bias=True):
        """
        Linear layer modified for equalized learning rate
        """
        from numpy import sqrt

        super().__init__()

        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in)
        ), requires_grad=True)

        self.use_bias = bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        return F.linear(
            x, self.weight * self.scale,
            self.bias if self.use_bias else None
        )
