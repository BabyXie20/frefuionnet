"""
基本卷积块模块
包含：ConvBlock、ResidualConvBlock、ResidualFuse3D 等基础构建块
"""

import torch
from torch import nn
import torch.nn.functional as F

from .utils import norm3d


# ========== 基础卷积块 ==========
class ConvBlock(nn.Module):
    """
    多阶段卷积块
    
    参数:
        n_stages: 卷积阶段数
        n_filters_in: 输入通道数
        n_filters_out: 输出通道数
        normalization: 归一化方式 ('batchnorm', 'groupnorm', 'instancenorm', 'none')
    
    结构:
        Conv3d(3x3x3) -> Norm -> ReLU
        Conv3d(3x3x3) -> Norm -> ReLU
        ... (重复 n_stages 次)
    
    例子:
        block = ConvBlock(2, 64, 128, normalization='instancenorm')
        out = block(x)  # [B, 128, D, H, W]
    """
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1, bias=False))
            ops.append(norm3d(normalization, n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class ResidualConvBlock(nn.Module):
    """
    多阶段残差卷积块
    
    参数:
        n_stages: 卷积阶段数
        n_filters_in: 输入通道数
        n_filters_out: 输出通道数
        normalization: 归一化方式
    
    结构:
        Conv3d(3x3x3) -> Norm -> ReLU
        Conv3d(3x3x3) -> Norm
        + 残差连接
        ReLU
    
    特点:
        - 支持跳跃连接（残差连接）
        - 最后一个卷积阶段后才应用 ReLU（符合 ResNet 设计）
        - 通道数可变
    
    例子:
        block = ResidualConvBlock(2, 64, 64, normalization='instancenorm')
        out = block(x)  # [B, 64, D, H, W]
    """
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1, bias=False))
            ops.append(norm3d(normalization, n_filters_out))
            if i != n_stages - 1:  # 最后一个阶段不添加激活，放到外面处理
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ResidualFuse3D(nn.Module):
    """
    3D 残差融合块 - 两层卷积 + 残差连接 + 可选通道投影
    
    参数:
        in_ch: 输入通道数
        out_ch: 输出通道数
        normalization: 归一化方式
    
    结构:
        ├─ Conv3d(3x3x3) -> Norm -> ReLU
        ├─ Conv3d(3x3x3) -> Norm
        ├─ [可选] 1x1x1 投影 (当 in_ch != out_ch 时)
        └─ 残差相加 -> ReLU
    
    特点:
        - 标准的两层残差块
        - 支持通道改变（通过 1x1x1 投影）
        - 适用于编码器/解码器的融合操作
    
    例子:
        # 通道不变
        block = ResidualFuse3D(128, 128, normalization='instancenorm')
        
        # 通道改变
        block = ResidualFuse3D(256, 128, normalization='instancenorm')
    """
    def __init__(self, in_ch, out_ch, normalization='none'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = norm3d(normalization, out_ch)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = norm3d(normalization, out_ch)

        # 通道投影：当输入输出通道不同时需要投影
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # 主分支
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # 恒等分支投影
        if self.proj is not None:
            identity = self.proj(identity)
        
        # 残差相加 + 激活
        out = out + identity
        out = self.act_out(out)
        return out


class FreqFuse3D(nn.Module):
    """
    频段注意力融合块 - 高低频融合 + 频段/通道注意力
    
    参数:
        channels: 特征通道数
        normalization: 归一化方式
        reduction: 通道注意力的压缩比
    
    输入:
        low: [B, C, D, H, W]      低频子带（LLL）
        highs: [B, C, 7, D, H, W] 高频子带（7个）
    
    输出:
        [B, C, D, H, W] 融合后的特征
    
    工作流程:
        1. 对每个高频子带做 GAP，生成子带特征 [B, 7]
        2. 通过 MLP 生成 band-wise 权重 [B, 7]
        3. 对 7 个高频子带加权融合
        4. 通过 1x1x1 Conv 聚合高频
        5. 与低频相加
        6. 通道注意力加权
        7. 残差细化
    
    特点:
        - 自适应频段权重（可学习）
        - 通道级注意力机制
        - 多尺度特征融合
    
    例子:
        fuse = FreqFuse3D(channels=256, normalization='instancenorm')
        out = fuse(low, highs)
    """
    def __init__(self, channels, normalization='instancenorm', reduction=4):
        super().__init__()
        self.channels = channels

        # 7 个高频子带聚合的 1x1x1 卷积
        self.high_agg_conv = nn.Conv3d(channels * 7, channels, kernel_size=1, bias=False)

        # 频段注意力：[B,7] -> [B,7] 的权重
        hidden = max(7 * 2, 8)
        self.band_mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 7),
            nn.Sigmoid()
        )

        # 通道注意力：SE-style
        mid_ch = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                  # [B, C, 1, 1, 1]
            nn.Conv3d(channels, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 融合后的局部残差细化
        self.refine = ResidualFuse3D(channels, channels, normalization=normalization)

    def forward(self, low, highs):
        """
        low: [B, C, D, H, W]
        highs: [B, C, 7, D, H, W]
        返回: [B, C, D, H, W]
        """
        B, C, S, D, H, W = highs.shape
        assert S == 7, f"Expect 7 high-frequency subbands, got {S}"

        # 1) 频段描述：全局平均池化
        band_desc = highs.mean(dim=(3, 4, 5))   # [B, C, 7]
        band_desc = band_desc.mean(dim=1)       # [B, 7]

        # 2) MLP 生成 band-wise 权重
        band_weights = self.band_mlp(band_desc)         # [B, 7]
        band_weights = band_weights.view(B, 1, S, 1, 1, 1)  # [B, 1, 7, 1, 1, 1]

        # 3) 对高频子带加权
        highs_weighted = highs * band_weights           # [B, C, 7, D, H, W]

        # 4) 聚合 7 个子带：reshape -> 1x1x1 Conv
        highs_reshaped = highs_weighted.view(B, C * S, D, H, W)  # [B, 7C, D, H, W]
        high_agg = self.high_agg_conv(highs_reshaped)            # [B, C, D, H, W]

        # 5) 与低频融合 + 通道注意力 + 残差细化
        base = low + high_agg                       # [B, C, D, H, W]
        ch_att = self.channel_att(base)             # [B, C, 1, 1, 1]
        fused = base * ch_att                       # [B, C, D, H, W]
        fused = self.refine(fused)                  # [B, C, D, H, W]
        
        return fused


class MultiScaleLocal3D(nn.Module):
    """
    多尺度深度可分离局部特征提取（不使用 DWT 高频）
    输入:  x [B,C,D,H,W]
    输出:  out [B,C,D,H,W]
    """
    def __init__(self, channels, normalization='instancenorm'):
        super().__init__()
        self.channels = channels

        # 1×1×1 Conv（通道混合）
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 3×3×3 depthwise + 1×1×1 pointwise
        self.conv3x3 = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3, padding=1,
                groups=channels, bias=False
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 空洞 3×3×3 depthwise + 1×1×1 pointwise
        self.conv3x3_dilate = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3, padding=2, dilation=2,
                groups=channels, bias=False
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 多尺度局部注意力
        self.local_att = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            norm3d(normalization, channels),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.dropout   = nn.Dropout3d(0.2)

    def forward(self, x):
        # 多尺度局部特征
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv3x3_dilate(x)

        concat_f = torch.cat([f1, f2, f3], dim=1)   # [B,3C,D,H,W]
        local_att = self.local_att(concat_f)        # [B,C,D,H,W]

        fused = (f1 + f2 + f3) * local_att
        out   = self.fuse_conv(fused)
        return self.dropout(out)


# ========== 导出接口 ==========
__all__ = [
    'ConvBlock',
    'ResidualConvBlock',
    'ResidualFuse3D',
    'FreqFuse3D',
    'MultiScaleLocal3D',
]
