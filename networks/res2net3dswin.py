"""
Res2Net 3D bottleneck for encoder feature extraction.
Keeps spatial size and replaces the first processed branch with a SwinTransformerV2
to inject global context (as in the paper's Fig.3 multi-scale branches).
"""

import math
import torch
from torch import nn

from .utils import norm3d
from .swin_transformer_v2 import SwinTransformerV2


class Res2NetBottleneck3D(nn.Module):
    """
    3D Res2Net bottleneck with a SwinTransformerV2 branch.

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels after the final 1x1x1 conv.
        scales (int): number of scale splits (s in the paper). Default: 4.
        base_width (int): base width used to compute per-scale width. Default: 26.
        img_size (tuple[int]): (D, H, W) for the Swin branch. Required.
        swin_window_size (tuple[int]): window size for Swin. Default: (4, 4, 4).
        swin_depths (tuple[int]): depths for Swin. Default: (1,).
        swin_heads (int | None): number of heads for Swin; if None, picked as a divisor of width.
        normalization (str): see utils.norm3d.
        activation (nn.Module): activation layer. Default: nn.ReLU(inplace=True).
    """

    def __init__(self, in_channels, out_channels, scales=4, base_width=26,
                 img_size=None, swin_window_size=(4, 4, 4), swin_depths=(1,),
                 swin_heads=None, normalization='instancenorm', activation=None):
        super().__init__()
        assert scales >= 2, "scales must be >= 2"
        assert img_size is not None, "img_size (D, H, W) must be provided for the Swin branch."
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

        # pick heads that divide width if not provided
        def pick_heads(embed_dim, preferred=4):
            if swin_heads is not None:
                return swin_heads
            head = preferred
            while head > 1 and embed_dim % head != 0:
                head -= 1
            return head

        # Swin branch replacing the first processed split (i == 1)
        self.swin = SwinTransformerV2(
            img_size=img_size,
            patch_size=(1, 1, 1),
            in_chans=width,
            num_classes=0,          # feature-only
            embed_dim=width,
            depths=swin_depths,
            num_heads=(pick_heads(width),),
            window_size=swin_window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=(0,) * len(swin_depths)
        )

        # other scale conv branches (3x3x3)
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
                if i == 1:
                    residual = spx[i]
                    # Swin branch keeps spatial size: run layers without avgpool/head
                    B, C, D, H, W = residual.shape
                    y_tokens = self.swin.patch_embed(residual)
                    if self.swin.ape:
                        y_tokens = y_tokens + self.swin.absolute_pos_embed
                    y_tokens = self.swin.pos_drop(y_tokens)
                    for layer in self.swin.layers:
                        y_tokens = layer(y_tokens)
                    y_tokens = self.swin.norm(y_tokens)
                    y = y_tokens.transpose(1, 2).view(B, C, D, H, W)
                else:
                    residual = spx[i] + outputs[-1]
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


__all__ = ["Res2NetBottleneck3D"]
