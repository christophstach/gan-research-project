import torch.nn as nn


class UpsampleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upscale_factor=2, padding_mode="zeros"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="nearest"
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            )
        )

    def forward(self, x):
        x = self.main(x)

        return x


class Conv2dPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upscale_factor=2, padding_mode="zeros"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2 ** upscale_factor,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                padding_mode=padding_mode
            ),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.main(x)

        return x
