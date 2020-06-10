import torch.nn as nn


# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha=1e-8):
        y = x ** 2
        y = y.mean(dim=1, keepdim=True)
        y = y + alpha
        y = y.sqrt()

        y = x / y
        return y
