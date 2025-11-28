import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .utils import norm3d, pairwise_cos_sim, pairwise_euclidean_sim, DWT3D, IDWT3D
from .blocks import ConvBlock, ResidualConvBlock, ResidualFuse3D, FreqFuse3D, MultiScaleLocal3D
from .FDW3D import FDWConv3d

class WaveletDown(nn.Module):
    """
    编码阶段：
    - body: ConvBlock / ResidualConvBlock → 得到 feat 作为 skip
    - DWT3D(feat) → low, highs
    - FreqFuse3D(low, highs) → fused (作为下一层输入)
    返回: next_input, highs, feat
        next_input : [B,out_ch,D/2,H/2,W/2]（编码器下一层输入）
        highs      : [B,out_ch,7,D/2,H/2,W/2]（高频子带，留给解码器做 IDWT）
        feat       : [B,out_ch,D,H,W]（未降采样 skip）
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        n_stages=2,
        normalization='none',
        has_residual=False,
        use_freq_fuse=True,
        freq_fuse_reduction=4,
        freq_fuse_norm='instancenorm'
    ):
        super().__init__()
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.body = convBlock(n_stages, in_ch, out_ch, normalization=normalization)
        self.dwt  = DWT3D()

        self.use_freq_fuse = use_freq_fuse
        if use_freq_fuse:
            self.fuse = FreqFuse3D(
                channels=out_ch,
                normalization=freq_fuse_norm,
                reduction=freq_fuse_reduction
            )
        else:
            self.fuse = None

    def forward(self, x):
        feat = self.body(x)                 # [B,out_ch,D,H,W]  (skip)
        low, highs = self.dwt(feat)         # low:[B,out_ch,D/2,H/2,W/2]; highs:[B,out_ch,7,D/2,H/2,W/2]

        if self.use_freq_fuse and self.fuse is not None:
            next_input = self.fuse(low, highs)  # [B,out_ch,D/2,H/2,W/2]
        else:
            next_input = low

        # next_input: 编码器下一层输入
        # highs    : 原始高频子带，留给解码器做 IDWT
        # feat     : skip feature（未降采样）
        return next_input, highs, feat


class WaveletUp(nn.Module):
    """
    解码阶段：
    - IDWT3D(low_from_decoder, highs_from_encoder) 上采样一倍
    - 与同层 skip_enc 拼接
    - ResidualFuse3D 融合
    """
    def __init__(self, in_ch_after_concat, out_ch, normalization='none'):
        super().__init__()
        self.idwt = IDWT3D()
        self.fuse = ResidualFuse3D(in_ch_after_concat, out_ch, normalization=normalization)

    def forward(self, low_from_decoder, highs_from_encoder, skip_enc):
        # IDWT 上采样一倍（通道不变）
        x = self.idwt(low_from_decoder, highs_from_encoder)  # [B,out_ch,D,H,W]
        # 与 skip 特征拼接
        x = torch.cat([x, skip_enc], dim=1)
        return self.fuse(x)


class BottleneckTransformerLite3D(nn.Module):
    """
    瓶颈层（精简版）：
    - 适用于已经经过 FreqFuse3D 融合高低频之后的特征
    - 结构：FDWConv3d(in) → 轻量 local 卷积 → 1 个 Transformer Block → FDWConv3d(out) → 残差

    输入 / 输出:
        x: [B, C, D, H, W]，其中 C = channels（比如 c5）
    """
    def __init__(
        self,
        channels: int,
        normalization: str = 'instancenorm',
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_light_local: bool = True,
    ):
        super().__init__()
        C = channels
        self.channels = C
        self.use_light_local = use_light_local

        # 1) 进瓶颈：频域动态卷积 + norm + ReLU
        self.conv_in = nn.Sequential(
            FDWConv3d(
                in_channels=C,
                out_channels=C,
                kernel_size=3,
                stride=1,
                padding=1,
                kernel_num=4,
                reduction=8
            ),
            norm3d(normalization, C),
            nn.ReLU(inplace=True)
        )

        # 2) 轻量 local 分支：一个 DW 3x3 + PW 1x1（可看作精简版 MultiScaleLocal3D）
        if use_light_local:
            self.local_conv = nn.Sequential(
                nn.Conv3d(
                    C, C,
                    kernel_size=3, padding=1,
                    groups=C, bias=False           # depthwise
                ),
                nn.Conv3d(C, C, kernel_size=1, bias=False),  # pointwise
                norm3d(normalization, C),
                nn.ReLU(inplace=True)
            )
        else:
            self.local_conv = None

        # 3) Transformer Block（标准 MHSA + FFN）
        hidden_dim = int(C * mlp_ratio)

        self.norm1 = nn.LayerNorm(C)
        self.attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            batch_first=True,   # 输入 [B, N, C]
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, C)
        )
        self.drop = nn.Dropout(dropout)

        # 4) 出瓶颈：再来一次 FDWConv3d（频域增强 + 通道重加权）
        self.conv_out = nn.Sequential(
            FDWConv3d(
                in_channels=C,
                out_channels=C,
                kernel_size=3,
                stride=1,
                padding=1,
                kernel_num=4,
                reduction=8
            ),
            norm3d(normalization, C),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        assert C == self.channels
        identity = x

        # Step 1: 频域动态卷积（已经融合好的多频特征再适配一下）
        x = self.conv_in(x)                              # [B,C,D,H,W]

        # Step 2: 轻量 local（可选，残差式叠加）
        if self.use_light_local and self.local_conv is not None:
            x = x + self.local_conv(x)

        # Step 3: 展平成 token 做 Transformer
        # [B,C,D,H,W] -> [B,N,C], N = D*H*W
        x_flat = x.view(B, C, -1).permute(0, 2, 1)       # [B,N,C]

        # MHSA + 残差
        h = self.norm1(x_flat)
        attn_out, _ = self.attn(h, h, h)                 # [B,N,C]
        x_flat = x_flat + self.drop(attn_out)

        # FFN + 残差
        h2 = self.norm2(x_flat)
        ffn_out = self.mlp(h2)                           # [B,N,C]
        x_flat = x_flat + self.drop(ffn_out)

        # 回到 3D
        x = x_flat.permute(0, 2, 1).view(B, C, D, H, W)  # [B,C,D,H,W]

        # Step 4: 再做一次 FDWConv3d + 残差
        x = self.conv_out(x)
        out = x + identity

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=14,
        n_filters=16,
        normalization='none',
        has_dropout=False,
        has_residual=False,
        use_freq_fuse=True,
        freq_fuse_reduction=4
    ):
        super().__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        c1, c2, c3, c4, c5 = n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16

        self.stem = convBlock(1, n_channels, c1, normalization=normalization)

        self.down1 = WaveletDown(
            c1, c2,
            n_stages=1,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down2 = WaveletDown(
            c2, c3,
            n_stages=2,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down3 = WaveletDown(
            c3, c4,
            n_stages=3,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down4 = WaveletDown(
            c4, c5,
            n_stages=3,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )

        self.bottleneck = BottleneckTransformerLite3D(
            channels=c5,
            normalization=normalization,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,        # 如果后面想再强一点正则，可以改成 0.1 左右
            use_light_local=True
        )


        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        x1 = self.stem(x)                 # [B,c1, D,H,W] (原始分辨率)

        l2, h2, s1 = self.down1(x1)       # s1: [B,c2, D/2?]     ; l2: [B,c2, 1/2]
        l3, h3, s2 = self.down2(l2)       # s2: [B,c3, 1/2]      ; l3: [B,c3, 1/4]
        l4, h4, s3 = self.down3(l3)       # s3: [B,c4, 1/4]      ; l4: [B,c4, 1/8]
        l5, h5, s4 = self.down4(l4)       # s4: [B,c5, 1/8]      ; l5: [B,c5, 1/16]
                                          # h5: [B,c5,7,1/16]
        z = self.bottleneck(l5)       # [B,c5, 1/16]

        if self.has_dropout:
            z = self.dropout(z)

        return {
            'skips': [s1, s2, s3, s4],    # 1, 1/2, 1/4, 1/8
            'highs': [h2, h3, h4, h5],    # 对称传给解码 IDWT
            'bottleneck': z
        }


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=14,
        n_filters=16,
        normalization='none',
        has_dropout=False
    ):
        super().__init__()
        self.has_dropout = has_dropout
        c1, c2, c3, c4, c5 = n_filters, n_filters * 2, n_filters * 4, n_filters * 8, n_filters * 16

        self.up4 = WaveletUp(in_ch_after_concat=c5 + c5, out_ch=c4, normalization=normalization)  # 1/16->1/8
        self.up3 = WaveletUp(in_ch_after_concat=c4 + c4, out_ch=c3, normalization=normalization)  # 1/8 ->1/4
        self.up2 = WaveletUp(in_ch_after_concat=c3 + c3, out_ch=c2, normalization=normalization)  # 1/4 ->1/2
        self.up1 = WaveletUp(in_ch_after_concat=c2 + c2, out_ch=c1, normalization=normalization)  # 1/2 ->1

        self.head_pre = ConvBlock(1, c1, c1, normalization=normalization)
        self.out_conv  = nn.Conv3d(c1, n_classes, kernel_size=1)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features: dict):
        s1, s2, s3, s4 = features['skips']
        h2, h3, h4, h5 = features['highs']
        z = features['bottleneck']

        x4 = self.up4(z,  h5, s4)  # 1/16 -> 1/8
        x3 = self.up3(x4, h4, s3)  # 1/8  -> 1/4
        x2 = self.up2(x3, h3, s2)  # 1/4  -> 1/2
        x1 = self.up1(x2, h2, s1)  # 1/2  -> 1

        x1 = self.head_pre(x1)
        if self.has_dropout:
            x1 = self.dropout(x1)
        out_seg = self.out_conv(x1)
        embedding = x1
        return out_seg, embedding


class VNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=14,
        patch_size=96,
        n_filters=16,
        normalization='instancenorm',
        has_dropout=False,
        has_residual=False,
        use_freq_fuse=True,
        freq_fuse_reduction=4
    ):
        super(VNet, self).__init__()
        self.num_classes = n_classes
        self.encoder = Encoder(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction
        )
        self.decoder = Decoder(
            n_channels=n_channels,
            n_classes=n_classes,
            n_filters=n_filters,
            normalization=normalization,
            has_dropout=has_dropout
        )

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, feat_dict):
        return self.decoder(feat_dict)

    def forward(self, x):
        feats = self.encoder(x)
        out_seg, embedding = self.decoder(feats)
        return out_seg, embedding
