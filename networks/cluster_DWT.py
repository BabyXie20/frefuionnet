import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .utils import norm3d, pairwise_cos_sim, pairwise_euclidean_sim, DWT3D, IDWT3D
from .blocks import ConvBlock, ResidualConvBlock, ResidualFuse3D, FreqFuse3D
from .res2net3dswin import Res2NetBottleneck3D


# ========== 小波上下采样块 ==========
class WaveletDown(nn.Module):
    """
    编码阶段：
    - body: ConvBlock / ResidualConvBlock → 得到 feat 作为 skip
    - DWT3D(feat) → low, highs
    - FreqFuse3D(low, highs) → fused (作为下一层输入)
    返回: fused_low, highs, feat
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        normalization='none',
        has_residual=False,
        use_freq_fuse=True,
        freq_fuse_reduction=4,
        freq_fuse_norm='instancenorm'
    ):
        super().__init__()
        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.body = convBlock(2, in_ch, out_ch, normalization=normalization)  # 作为 skip
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
        low, highs = self.dwt(feat)         # low:[B,out_ch,D/2,H/2,W/2]; highs:[B,out_ch,7,...]

        if self.use_freq_fuse and self.fuse is not None:
            next_input = self.fuse(low, highs)  # [B,out_ch,D/2,H/2,W/2]
        else:
            next_input = low

        # next_input: 编码器下一层输入
        # highs    : 原始高频子带，留给解码器做 IDWT
        # feat     : skip feature（未降采样）
        return next_input, highs, feat


class WaveletUp(nn.Module):
    def __init__(self, in_ch_after_concat, out_ch, normalization='none'):
        super().__init__()
        self.idwt = IDWT3D()
        self.fuse = ResidualFuse3D(in_ch_after_concat, out_ch, normalization=normalization)

    def forward(self, low_from_decoder, highs_from_encoder, skip_enc):
        x = self.idwt(low_from_decoder, highs_from_encoder)  # 通道不变，上采样一倍
        x = torch.cat([x, skip_enc], dim=1)
        return self.fuse(x)


class Cluster3DVolume(nn.Module):
    """
    3D 聚类稀疏注意力（思路2版：支持 Top-k 软聚类）

    输入:
        x: [B, C, D, H, W]

    参数:
        dim        : 输入通道数 C
        out_dim    : 输出通道数（一般设成和 dim 相同）
        heads      : 多头数
        head_dim   : 每个 head 的通道数，需要满足 dim = heads * head_dim
        proposal_d/h/w : AdaptiveAvgPool3d 后簇中心网格大小 (Dp,Hp,Wp)，K=Dp*Hp*Wp
        fold_d/h/w : 可选，把大体积切成 3D block 再做聚类
        return_center  : True 时返回中心特征 (Dp×Hp×Wp)；False 时还原成 D×H×W
        topk      : 若为 None，则对所有 K 个簇 softmax；
                    若为整数 < K，则每个 voxel 只与 top-k 个簇连接（soft assignment）
    """
    def __init__(self,
                 dim=256,
                 out_dim=256,
                 heads=4,
                 head_dim=64,
                 proposal_d=2,
                 proposal_h=2,
                 proposal_w=2,
                 fold_d=1,
                 fold_h=1,
                 fold_w=1,
                 return_center: bool = False,
                 topk: int = None):
        super().__init__()
        assert dim == heads * head_dim, "dim 必须等于 heads * head_dim"

        self.heads = heads
        self.head_dim = head_dim
        self.return_center = return_center
        self.fold_d = fold_d
        self.fold_h = fold_h
        self.fold_w = fold_w
        self.topk = topk

        # f / v / proj 改成 Conv3d
        self.f = nn.Conv3d(dim, heads * head_dim, kernel_size=1)
        self.v = nn.Conv3d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv3d(heads * head_dim, out_dim, kernel_size=1)

        # 相似度缩放参数
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        # 3D Adaptive Pooling 生成簇中心
        self.centers_proposal = nn.AdaptiveAvgPool3d((proposal_d, proposal_h, proposal_w))

        self.rule1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        # 1) f / v 投影
        value = self.v(x)  # [B, E*C_h, D,H,W]
        feat  = self.f(x)  # [B, E*C_h, D,H,W]

        # 2) 拆多头
        feat = rearrange(feat,  "b (e c) d h w -> (b e) c d h w", e=self.heads)
        value = rearrange(value, "b (e c) d h w -> (b e) c d h w", e=self.heads)

        # 3) 可选 fold
        if self.fold_d > 1 or self.fold_h > 1 or self.fold_w > 1:
            b0, c0, d0, h0, w0 = feat.shape
            assert d0 % self.fold_d == 0 and h0 % self.fold_h == 0 and w0 % self.fold_w == 0, \
                f"feature size ({d0},{h0},{w0}) 不能被 fold ({self.fold_d},{self.fold_h},{self.fold_w}) 整除"
            feat = rearrange(
                feat,
                "b c (fd d) (fh h) (fw w) -> (b fd fh fw) c d h w",
                fd=self.fold_d, fh=self.fold_h, fw=self.fold_w
            )
            value = rearrange(
                value,
                "b c (fd d) (fh h) (fw w) -> (b fd fh fw) c d h w",
                fd=self.fold_d, fh=self.fold_h, fw=self.fold_w
            )

        b1, c1, d1, h1, w1 = feat.shape
        N = d1 * h1 * w1

        # 4) 3D pooling 得到簇中心
        centers = self.centers_proposal(feat)        # [b1, c1, Dp,Hp,Wp]
        value_centers = self.centers_proposal(value) # [b1, c1, Dp,Hp,Wp]
        Dp, Hp, Wp = centers.shape[2:]
        K = Dp * Hp * Wp

        # 5) flatten
        centers_flat = centers.view(b1, c1, K).permute(0, 2, 1)  # [b1, K, c1]
        tokens_flat  = feat.view(b1, c1, N).permute(0, 2, 1)     # [b1, N, c1]

        # 6) 相似度
        sim_raw = pairwise_cos_sim(centers_flat, tokens_flat)    # [b1, K, N]
        sim = self.rule1(self.sim_beta + self.sim_alpha * sim_raw)

        # 7) top-k 稀疏
        if (self.topk is not None) and (self.topk < K):
            k = self.topk
            topk_val, topk_idx = torch.topk(sim, k=k, dim=1)     # [b1, k, N]
            sim_masked = sim.new_full(sim.shape, float('-inf'))
            sim_masked.scatter_(1, topk_idx, topk_val)
            sim = sim_masked

        # 8) softmax 注意力
        attn = F.softmax(sim, dim=1)          # [b1, K, N]

        value_flat = value.view(b1, c1, N).permute(0, 2, 1)  # [b1, N, c1]

        # 9) 成员 -> 中心
        cluster_sum = torch.einsum('bkn,bnc->bkc', attn, value_flat)  # [b1, K, c1]
        denom = attn.sum(dim=2, keepdim=True).clamp(min=1e-6)         # [b1, K,1]
        cluster_feat = cluster_sum / denom

        value_centers_flat = value_centers.view(b1, c1, K).permute(0, 2, 1)  # [b1,K,c1]
        cluster_feat = cluster_feat + value_centers_flat                     # [b1,K,c1]

        if self.return_center:
            out = cluster_feat.permute(0, 2, 1).contiguous().view(b1, c1, Dp, Hp, Wp)
        else:
            # 10) 中心 -> 成员
            out_tokens = torch.einsum('bkn,bkc->bnc', attn, cluster_feat)    # [b1, N, c1]
            out = out_tokens.permute(0, 2, 1).contiguous().view(b1, c1, d1, h1, w1)

        # 11) fold 还原
        if self.fold_d > 1 or self.fold_h > 1 or self.fold_w > 1:
            out = rearrange(
                out,
                "(b fd fh fw) c d h w -> b c (fd d) (fh h) (fw w)",
                fd=self.fold_d, fh=self.fold_h, fw=self.fold_w
            )

        # 12) 合并 heads + 1x1x1 proj
        out = rearrange(out, "(b e) c d h w -> b (e c) d h w", e=self.heads)
        out = self.proj(out)
        return out


# ========== 条纹卷积注意力块 ==========
class StripeConvAttention3D(nn.Module):
    """
    第三分支：条状卷积 + 条状注意力
    - 3 个 depthwise 条形卷积：
        conv_d: k x 1 x 1  (沿 D 轴)
        conv_h: 1 x k x 1  (沿 H 轴)
        conv_w: 1 x 1 x k  (沿 W 轴)
    - 3 个轴向注意力：
        对输入 x 在对应两个维度上做平均，得到条状描述，再用 1x1x1 Conv + Sigmoid 得到注意力图

    输入:  x [B,C,D,H,W]
    输出:  out [B,C,D,H,W]
    """
    def __init__(self, channels, kernel_size=3, normalization='instancenorm'):
        super().__init__()
        self.channels = channels
        k = kernel_size
        pad = k // 2

        # 深度可分离条形卷积（每个通道独立）
        self.conv_d = nn.Conv3d(
            channels, channels,
            kernel_size=(k, 1, 1),
            padding=(pad, 0, 0),
            groups=channels,
            bias=False
        )
        self.conv_h = nn.Conv3d(
            channels, channels,
            kernel_size=(1, k, 1),
            padding=(0, pad, 0),
            groups=channels,
            bias=False
        )
        self.conv_w = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 1, k),
            padding=(0, 0, pad),
            groups=channels,
            bias=False
        )

        # 条状注意力：三个轴向各一个 1x1x1 Conv
        self.att_d = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.att_h = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.att_w = nn.Conv3d(channels, channels, kernel_size=1, bias=True)

        # 融合三个方向的通道：3C -> C
        self.pointwise = nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False)
        self.norm = norm3d(normalization, channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,D,H,W]
        """
        B, C, D, H, W = x.shape

        # 1) 条形卷积（directional conv）
        x_d = self.conv_d(x)   # [B,C,D,H,W]
        x_h = self.conv_h(x)   # [B,C,D,H,W]
        x_w = self.conv_w(x)   # [B,C,D,H,W]

        # 2) 条状注意力（axis-wise）
        # D轴：对 H,W 求平均 -> [B,C,D,1,1]
        d_desc = x.mean(dim=(3, 4), keepdim=True)
        att_d  = torch.sigmoid(self.att_d(d_desc))      # [B,C,D,1,1]

        # H轴：对 D,W 求平均 -> [B,C,1,H,1]
        h_desc = x.mean(dim=(2, 4), keepdim=True)
        att_h  = torch.sigmoid(self.att_h(h_desc))      # [B,C,1,H,1]

        # W轴：对 D,H 求平均 -> [B,C,1,1,W]
        w_desc = x.mean(dim=(2, 3), keepdim=True)
        att_w  = torch.sigmoid(self.att_w(w_desc))      # [B,C,1,1,W]

        # 3) 用轴向注意力调制对应条形卷积输出
        x_d = x_d * att_d
        x_h = x_h * att_h
        x_w = x_w * att_w

        # 4) 融合三个方向
        x_cat = torch.cat([x_d, x_h, x_w], dim=1)       # [B,3C,D,H,W]
        out   = self.pointwise(x_cat)                   # [B,C,D,H,W]
        out   = self.norm(out)
        out   = self.act(out)
        return out


class BottleneckCluster3D(nn.Module):
    """
    瓶颈结构（三分支）：
        串联：
            conv_in(3x3x3) -> base

        并联三支：
            1) local_branch   : ResidualFuse3D，本地卷积增强
            2) cluster_branch : Cluster3DVolume，聚类稀疏注意力（全局）
            3) stripe_branch  : StripeConvAttention3D，条状卷积 + 条状注意力

        分支级 + 通道级注意力融合：
            对 [local, cluster, stripe] 三条分支做 “per-channel softmax”：
            每个通道 C 有 (α_local, α_cluster, α_stripe)，三者和为 1

        串联：
            conv_out(3x3x3)(fused)

        最终：
            out = conv_out(fused) + x        # 残差
    """
    def __init__(
        self,
        channels: int,
        normalization: str = 'instancenorm',
        heads: int = 4,
        proposal_d: int = 2,
        proposal_h: int = 2,
        proposal_w: int = 2,
        topk: int = 3
    ):
        super().__init__()
        c5 = channels
        self.channels = c5

        # 1) 前置 3x3x3 conv（串联入口）
        self.conv_in = nn.Sequential(
            nn.Conv3d(c5, c5, kernel_size=3, padding=1, bias=False),
            norm3d(normalization, c5),
            nn.ReLU(inplace=True)
        )

        # 2) 本地分支：ResidualFuse3D
        self.local_branch = ResidualFuse3D(c5, c5, normalization=normalization)

        # 3) 聚类稀疏注意力分支
        head_dim = c5 // heads
        assert head_dim * heads == c5, "channels 必须能被 heads 整除"

        self.cluster_branch = Cluster3DVolume(
            dim=c5,
            out_dim=c5,
            heads=heads,
            head_dim=head_dim,
            proposal_d=proposal_d,
            proposal_h=proposal_h,
            proposal_w=proposal_w,
            fold_d=1,
            fold_h=1,
            fold_w=1,
            topk=topk
        )

        # 4) 第三分支：条状卷积 + 条状注意力
        self.stripe_branch = StripeConvAttention3D(
            channels=c5,
            kernel_size=3,
            normalization=normalization
        )

        # 5) 三分支的“分支×通道”注意力：
        #    输入 [B,3,C] 展平成 [B,3C]，MLP 输出 [B,3C]，再 reshape 成 [B,3,C]
        hidden = max(c5, 16)  #256
        self.branch_mlp = nn.Sequential(
            nn.Linear(3 * c5, hidden),   #[B,3C] -> [B,hidden]
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3 * c5) #[B,hidden] -> [B,3C]
        )

        # 6) 输出 3x3x3 conv（串联出口）
        self.conv_out = nn.Sequential(
            nn.Conv3d(c5, c5, kernel_size=3, padding=1, bias=False),
            norm3d(normalization, c5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,D,H,W]
        """
        B, C, D, H, W = x.shape
        assert C == self.channels

        # 串联 step 1：前置卷积
        base = self.conv_in(x)                         # [B,C,D,H,W]

        # 三条分支（并联）
        feat_local   = self.local_branch(base)         # [B,C,D,H,W]
        feat_cluster = self.cluster_branch(base)       # [B,C,D,H,W]
        feat_stripe  = self.stripe_branch(base)        # [B,C,D,H,W]

        # 堆叠为 [B,3,C,D,H,W]
        stack = torch.stack([feat_local, feat_cluster, feat_stripe], dim=1)

        # 对空间做 GAP 得到 [B,3,C]
        gap = stack.mean(dim=(3, 4, 5))                # [B,3,C]

        # MLP 生成 branch logits： [B,3C] -> [B,3,C]
        g = gap.view(B, 3 * C)                         # [B,3C]
        g = self.branch_mlp(g)                         # [B,3C]
        g = g.view(B, 3, C)                            # [B,3,C]

        # 在分支维度做 softmax，得到每个通道对应三分支的权重
        alpha = F.softmax(g, dim=1)                    # [B,3,C]

        # 扩展为 [B,3,C,1,1,1]，在整个空间共享这个通道级分支权重
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 加权融合三条分支 → [B,C,D,H,W]
        fused = (stack * alpha).sum(dim=1)

        # 串联 step 2：输出卷积
        out = self.conv_out(fused)                     # [B,C,D,H,W]

        # 残差：加回原始输入 x
        return out + x


# ------------------------------
# Encoder (Wavelet + 频段注意力)
# ------------------------------
class Encoder(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=14,
        n_filters=16,
        patch_size=96,
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

        def _to_3tuple(x):
            return x if isinstance(x, (tuple, list)) else (x, x, x)

        pD, pH, pW = _to_3tuple(patch_size)
        # stage3 输入分辨率：patch_size / 4， stage4 输入分辨率：patch_size / 8
        stage3_size = (pD // 4, pH // 4, pW // 4)
        stage4_size = (pD // 8, pH // 8, pW // 8)

        self.stem = convBlock(1, n_channels, c1, normalization=normalization)

        self.down1 = WaveletDown(
            c1, c2,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down2 = WaveletDown(
            c2, c3,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down3 = WaveletDown(
            c3, c4,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )
        self.down4 = WaveletDown(
            c4, c5,
            normalization=normalization,
            has_residual=has_residual,
            use_freq_fuse=use_freq_fuse,
            freq_fuse_reduction=freq_fuse_reduction,
            freq_fuse_norm=normalization
        )

        # Replace stage3/4 bodies with Res2NetBottleneck3D (keep spatial size)
        self.down3.body = Res2NetBottleneck3D(
            in_channels=c3,
            out_channels=c4,
            scales=4,
            base_width=26,
            img_size=stage3_size,
            swin_window_size=(4, 4, 4),
            swin_depths=(1,),
            normalization=normalization
        )
        self.down4.body = Res2NetBottleneck3D(
            in_channels=c4,
            out_channels=c5,
            scales=4,
            base_width=26,
            img_size=stage4_size,
            swin_window_size=(4, 4, 4),
            swin_depths=(1,),
            normalization=normalization
        )

        self.bottleneck = BottleneckCluster3D(
            channels=c5,
            normalization=normalization,
            heads=4,         
            proposal_d=2,     
            proposal_h=2,
            proposal_w=2,
            topk=3
        )

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        x1 = self.stem(x)                 # [B,c1, 1]

        l2, h2, s1 = self.down1(x1)       # s1: [B,c2, 1]     ; l2: [B,c2, 1/2]
        l3, h3, s2 = self.down2(l2)       # s2: [B,c3, 1/2]  ; l3: [B,c3, 1/4]
        l4, h4, s3 = self.down3(l3)       # s3: [B,c4, 1/4]  ; l4: [B,c4, 1/8]
        l5, h5, s4 = self.down4(l4)       # s4: [B,c5, 1/8]  ; l5: [B,c5, 1/16]

        z = self.bottleneck(l5)           # [B,c5, 1/16]

        if self.has_dropout:
            z = self.dropout(z)

        return {
            'skips': [s1, s2, s3, s4],    # 1, 1/2, 1/4, 1/8
            'highs': [h2, h3, h4, h5],    # 对称传给解码 IDWT
            'bottleneck': z
        }


# ------------------------------
# Decoder (Wavelet)
# ------------------------------
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
            patch_size=patch_size,
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
