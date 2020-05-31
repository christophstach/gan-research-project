import math

import torch
import torch.nn as nn

import gans.building_blocks as bb


def snn_weight_init(m):
    # Self-Normalizing Neural Networks
    # https://arxiv.org/abs/1706.02515
    if isinstance(m, nn.Conv2d) or isinstance(m, bb._Conv2d) or isinstance(m, nn.Linear):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = math.sqrt(1.0 / fan_in)
        bound = math.sqrt(1.0 / fan_in)
        torch.nn.init.normal_(m.weight, mean=0, std=std)

        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -bound, bound)

    if isinstance(m, bb._ConvTranspose2d):
        fan_in = m.in_channels
        std = math.sqrt(1.0 / fan_in)
        bound = math.sqrt(1.0 / fan_in)
        torch.nn.init.normal_(m.weight, mean=0, std=std)

        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -bound, bound)


def he_weight_init(m):
    # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    # https://arxiv.org/abs/1502.01852
    gain = torch.nn.init.calculate_gain("leaky_relu", param=0.2)

    if isinstance(m, nn.Conv2d) or isinstance(m, bb._Conv2d) or isinstance(m, bb._ConvTranspose2d):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = math.sqrt(gain / fan_in)
        bound = gain * math.sqrt(3.0 / fan_in)
        torch.nn.init.normal_(m.weight, mean=0, std=std)

        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -bound, bound)

    if isinstance(m, bb._ConvTranspose2d) or isinstance(m, nn.ConvTranspose2d):
        fan_in = m.in_channels
        std = math.sqrt(gain / fan_in)
        bound = gain * math.sqrt(3.0 / fan_in)
        torch.nn.init.normal_(m.weight, mean=0, std=std)

        if m.bias is not None:
            torch.nn.init.uniform_(m.bias, -bound, bound)

