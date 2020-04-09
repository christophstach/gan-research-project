import torch
import torch.nn as nn


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, k=8, bias=False):
        super().__init__()

        self.wf = nn.Conv2d(in_channels, in_channels // k, kernel_size=1, stride=1, padding=0, bias=bias)
        self.wg = nn.Conv2d(in_channels, in_channels // k, kernel_size=1, stride=1, padding=0, bias=bias)
        self.wh = nn.Conv2d(in_channels, in_channels // k, kernel_size=1, stride=1, padding=0, bias=bias)
        self.wv = nn.Conv2d(in_channels // k, in_channels, kernel_size=1, stride=1, padding=0, bias=bias)

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        f = self.wf(x).view(x.size(0), -1, x.size(2) * x.size(3))
        g = self.wg(x).view(x.size(0), -1, x.size(2) * x.size(3))
        h = self.wh(x).view(x.size(0), -1, x.size(2) * x.size(3))
        s = torch.bmm(f.transpose(1, 2), g)
        beta = torch.softmax(s, 2)

        v = torch.bmm(h, beta).view(x.size(0), -1, x.size(2), x.size(3))
        o = self.wv(v)

        return self.gamma * o + x
