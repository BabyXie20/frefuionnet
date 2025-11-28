"""
Res2Net 3D bottleneck (encoder-friendly) with all branches using 3x3x3 convs.
This version keeps spatial size (stride=1) and uses 3D convolutions for the first
processed split, instead of SwinTransformer.
"""

import math
import torch
from torch import nn

from .utils import norm3d


class Res2NetBottleneck3DConvFirst(nn.Module):
    """
    3D Res2Net bottleneck (no Swin branch, conv on first processed split).

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels after the final 1x1x1 conv.
        scales (int): number of scale splits (s in the paper). Default: 4.
        base_width (int): base width used to compute per-scale width. Default: 26.
        normalization (str): see utils.norm3d.
        activation (nn.Module): activation layer. Default: nn.ReLU(inplace=True).
    """

    def __init__(self, in_channels, out_channels, scales=4, base_width=26,
                 normalization='instancenorm', activation=None):
        super().__init__()
        assert scales >= 2, "scales must be >= 2"
        self.scales = scales
        self.activation = activation or nn.ReLU(inplace=True)
        self.stride = (1, 1, 1)  # keep spatial size

        # width per scale (same rule as original Res2Net)
        width = int(math.floor(out_channels * base_width / 64.0))
        self.width = width
        inner_channels = width * scales

        # stem 1x1x1
        self.conv1 = nn.Conv3d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn1 = norm3d(normalization, inner_channels)

        # scale conv branches (3x3x3) for processed splits (i >= 1)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(scales - 1):
            self.convs.append(nn.Conv3d(width, width, kernel_size=3, stride=1, padding=1, bias=False))
            self.bns.append(norm3d(normalization, width))

        # final fuse 1x1x1
        self.conv3 = nn.Conv3d(inner_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm3d(normalization, out_channels)

        # projection if channel count changes
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                norm3d(normalization, out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # split into scales
        spx = torch.split(out, self.width, dim=1)
        outputs = []
        for i in range(self.scales):
            if i == 0:
                outputs.append(spx[i])
            else:
                residual = spx[i] if i == 1 else spx[i] + outputs[-1]
                y = self.convs[i - 1](residual)
                y = self.bns[i - 1](y)
                y = self.activation(y)
                outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.activation(out)
        return out


__all__ = ["Res2NetBottleneck3DConvFirst"]
