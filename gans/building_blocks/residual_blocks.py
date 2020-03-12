import torch.nn as nn


class ResidualBlockTypeA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode="zero"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.last = nn.PReLU(out_channels)

    def forward(self, x):
        identity = x
        x = self.main(x)

        return self.last(identity + x)


class ResidualBlockTypeB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode="zero"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.main(x)

        return identity + x


class ResidualBlockTypeC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode="zero"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.PReLU(out_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            )
        )

    def forward(self, x):
        identity = x
        x = self.main(x)

        return identity + x
