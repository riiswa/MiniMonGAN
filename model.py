from typing import Optional, List

import torch
import torch.nn as nn
from torchvision.io import ImageReadMode


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        normalize: bool = True,
        dropout: bool = True,
        output_size: Optional[List[int]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.normalize = normalize
        self.dropout = dropout
        self.output_size = output_size
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels)
        if dropout:
            self.dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, self.output_size)
        if self.normalize:
            x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = DownsamplingBlock(
            in_channels=4,
            out_channels=64,
            kernel_size=4,
            stride=3,
            padding=1,
            normalize=False,
        )
        self.conv2 = DownsamplingBlock(
            in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1
        )
        self.conv3 = DownsamplingBlock(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1
        )
        self.conv4 = DownsamplingBlock(
            in_channels=256, out_channels=512, kernel_size=2, stride=2
        )
        self.conv5 = DownsamplingBlock(
            in_channels=512, out_channels=512, kernel_size=2, stride=2
        )
        self.conv6 = DownsamplingBlock(
            in_channels=512, out_channels=512, kernel_size=1, stride=2, normalize=False
        )

        self.t_conv1 = UpsamplingBlock(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            stride=2,
            normalize=False,
            output_size=[2, 2],
        )
        self.t_conv2 = UpsamplingBlock(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.t_conv3 = UpsamplingBlock(
            in_channels=1024, out_channels=256, kernel_size=2, stride=2
        )
        self.t_conv4 = UpsamplingBlock(
            in_channels=512, out_channels=128, kernel_size=2, stride=2, dropout=False
        )
        self.t_conv5 = UpsamplingBlock(
            in_channels=256, out_channels=64, kernel_size=2, stride=2, dropout=False
        )

        self.t_conv6 = UpsamplingBlock(
            in_channels=128, out_channels=32, kernel_size=2, stride=2, dropout=False
        )

        self.tanh = nn.Tanh()
        self.output_conv = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.avg_pooling = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        skip2 = x
        x = self.conv2(x)
        skip3 = x
        x = self.conv3(x)
        skip4 = x
        x = self.conv4(x)
        skip5 = x
        x = self.conv5(x)
        skip6 = x
        x = self.conv6(x)

        # UpSampling
        x = torch.cat((skip6, self.t_conv1(x)), 1)
        x = torch.cat((skip5, self.t_conv2(x)), 1)
        x = torch.cat((skip4, self.t_conv3(x)), 1)
        x = torch.cat((skip3, self.t_conv4(x)), 1)
        x = torch.cat((skip2, self.t_conv5(x)), 1)
        x = self.t_conv6(x)

        x = self.tanh(self.output_conv(x))
        x = self.avg_pooling(x)
        x = x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

        return x


class Discriminator(nn.Module):
    def __init__(self, patch_size: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.conv0 = DownsamplingBlock(
            in_channels=4,
            out_channels=4,
            kernel_size=4,
            stride=2,
            padding=17,
            normalize=False,
        )
        self.conv1 = DownsamplingBlock(
            in_channels=8,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            normalize=False,
        )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        source = self.conv0(source)
        x = torch.cat((x, source), 1)
        x = self.conv1(x)
        x = self.sigmoid(self.conv2(x))
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(2, 3, 0, 1, 4, 5).reshape(-1, 1, self.patch_size, self.patch_size)
        return x
