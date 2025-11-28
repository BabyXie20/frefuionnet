import torch
import torch.nn as nn

__all__ = ["TransformerBlock_3D_LKA", "LKA3d", "LKA_Attention3d"]


class ResidualConv3d(nn.Module):
    """A minimal 3D residual block without external dependencies."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.act(out + residual)
        return out


class TransformerBlock_3D_LKA(nn.Module):
    """Transformer-style block that mixes token and local LKA attention for 3D inputs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = False,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = LKA_Attention3d(d_model=hidden_size)
        self.conv51 = ResidualConv3d(hidden_size, hidden_size, kernel_size=3, stride=1)
        self.conv8 = nn.Sequential(
            nn.Dropout3d(dropout_rate, inplace=False) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(hidden_size, hidden_size, 1),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        x_tokens = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x_tokens = x_tokens + self.pos_embed
        attn = x_tokens + self.gamma * self.epa_block(self.norm(x_tokens), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        attn = self.conv51(attn_skip)
        out = attn_skip + self.conv8(attn)
        return out


class LKA3d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        kernel_dwd = 7
        dilation_dwd = 3
        padding_dwd = 9
        kernel_dw = 5
        padding_dw = 2

        self.conv0 = nn.Conv3d(dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim)
        self.conv_spatial = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_dwd,
            stride=1,
            padding=padding_dwd,
            groups=dim,
            dilation=dilation_dwd,
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LKA_Attention3d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor, B: int, C: int, H: int, W: int, D: int) -> torch.Tensor:
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
