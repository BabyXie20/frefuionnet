"""
Thin wrapper around MONAI's SwinTransformer to extract global features
while keeping the spatial resolution unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep


class KeepSizePatchMerging(nn.Module):
    """
    Drop-in replacement for MONAI's PatchMerging that only expands channels
    (x2) and preserves spatial dimensions.
    """

    def __init__(self, dim: int, norm_layer: type[nn.Module] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, 2 * dim, bias=False)
        self.spatial_dims = spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, d, h, w, c = x.shape
            x = x.view(b * d * h * w, c)
            x = self.norm(x)
            x = self.reduction(x)
            x = x.view(b, d, h, w, -1)
        elif x.dim() == 4:
            b, h, w, c = x.shape
            x = x.view(b * h * w, c)
            x = self.norm(x)
            x = self.reduction(x)
            x = x.view(b, h, w, -1)
        else:
            raise ValueError(f"KeepSizePatchMerging expects 4D/5D input, got shape {tuple(x.shape)}")
        return x


class SwinGlobalFeature(nn.Module):
    """
    SwinTransformer backbone that keeps input spatial size and outputs a global
    feature map (final stage) or the full multi-scale list from MONAI.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = (4, 4, 4),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        out_channels: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        assert len(depths) == 4 and len(num_heads) == 4, "MONAI SwinTransformer expects 4 stages."

        patch_size = ensure_tuple_rep(1, spatial_dims)  # keep spatial size
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        self.backbone = SwinTransformer(
            in_chans=in_channels,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=KeepSizePatchMerging,
            use_v2=False,
        )

        conv_cls = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        final_channels = embed_dim * (2 ** self.backbone.num_layers)
        self.head = conv_cls(final_channels, out_channels, kernel_size=1) if out_channels else nn.Identity()
        self.normalize_outputs = normalize

    def forward(self, x: torch.Tensor, return_multi_scale: bool = False) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            x: input tensor [B, C, D, H, W] or [B, C, H, W].
            return_multi_scale: if True, return all Swin feature maps (x0..x4);
                otherwise only return the final global feature.
        """
        feats = self.backbone(x, normalize=self.normalize_outputs)
        if return_multi_scale:
            return feats
        return self.head(feats[-1])


__all__ = ["KeepSizePatchMerging", "SwinGlobalFeature"]
