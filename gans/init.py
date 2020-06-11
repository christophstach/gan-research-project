import math

import torch
import torch.nn as nn

import gans.building_blocks as bb

from math import sqrt

def _shared_weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.constant_(m.bias, 0)
        

def selu_weight_init(m):
    _shared_weight_init(m)

    # Self-Normalizing Neural Networks
    # https://arxiv.org/abs/1706.02515
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        nn.init.normal_(m.weight, 0, sqrt(1. / fan_in))  
    elif isinstance(m, nn.Linear):
        fan_in = m.in_features
        nn.init.normal_(m.weight, 0, sqrt(1. / fan_in))


def he_weight_init(m):
    _shared_weight_init(m)

    # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    # https://arxiv.org/abs/1502.01852
    gain = torch.nn.init.calculate_gain("leaky_relu", param=0.2)

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.2)

def orthogonal_weight_init(m):
    _shared_weight_init(m)

    gain = torch.nn.init.calculate_gain("leaky_relu", param=0.2)

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=gain)