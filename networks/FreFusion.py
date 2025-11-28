import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft


def build_3d_freq_masks(kd, kh, kw, kernel_num, device=None, dtype=torch.bool):
    """
    在 (kd,kh,kw) 的 3D 频域上构建 kernel_num 个互不重叠的频率掩码，
    按频率能量的分位数从低到高划分。
    返回: [K, kd, kh, kw] 的 bool 张量。
    """
    fd = torch.fft.fftfreq(kd, device=device)
    fh = torch.fft.fftfreq(kh, device=device)
    fw = torch.fft.fftfreq(kw, device=device)

    FZ, FY, FX = torch.meshgrid(fd, fh, fw, indexing='ij')
    radius = torch.sqrt(FZ ** 2 + FY ** 2 + FX ** 2)

    energy = radius ** 2
    energy = energy.flatten()

    quantiles = torch.linspace(0, 1, kernel_num + 1, device=device)
    bounds = torch.quantile(energy, quantiles, interpolation='linear')

    masks = []
    for i in range(kernel_num):
        low, high = bounds[i], bounds[i + 1]
        # 若担心相邻 band 边界重叠，可把 energy >= low 改成 energy > low (i>0 时)
        if i == 0:
            mask = (energy >= low) & (energy <= high)
        else:
            mask = (energy > low) & (energy <= high)
        mask = mask.view(kd, kh, kw)
        masks.append(mask)

    masks = torch.stack(masks, dim=0)
    return masks.to(dtype)


class FDWConv3d(nn.Module):
    """
    3D 频率动态卷积（FDWConv3d）+ FBM + KSM（kernel 级别 element-wise 调制）

    特点:
    - 频域参数 + 互不重叠的频带掩码 (FDW)
    - 空间版频带 gate (FBM 简化 3D 版)：每个 voxel 对 K 个频带做 softmax
    - Kernel-level Spatial Modulation (KSM)：
        针对 [K, Cout, Cin, kd, kh, kw] 中每个权重元素，学习独立的缩放因子 α
        做 element-wise 调制

    保持接口与普通 Conv3d 类似，可无缝替换 3x3 卷积。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        kernel_num=4,
        reduction=4,
    ):
        super().__init__()
        # 支持非立方核
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        kd, kh, kw = kernel_size
        self.kd, self.kh, self.kw = kd, kh, kw
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_num = kernel_num
        self.reduction = max(4, min(8, in_channels // 64))  # 智能设置 reduction

        # 1) 频域参数：[Cout, Cin, kd, kh, kw, 2] (最后一维为实部/虚部)
        self.freq_param = nn.Parameter(
            torch.randn(out_channels, in_channels, kd, kh, kw, 2) * 1e-3
        )

        # 2) 频率掩码（延迟初始化）: [K,1,1,kd,kh,kw]
        self.register_buffer("freq_masks", None, persistent=False)

        # 3) 空间版频带 gate（FBM）：每个 voxel 对 K 个频带做 softmax
        band_hidden = max(in_channels // self.reduction, 4)
        self.band_spatial = nn.Sequential(
            nn.Conv3d(in_channels, band_hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(band_hidden, kernel_num, kernel_size=1, bias=True)
        )
        self.band_softmax = nn.Softmax(dim=1)

        # 4) KSM：针对每个频带、每个卷积核元素的 element-wise 调制
        #    形状与空间域 kernel 完全一致：[K, Cout, Cin, kd, kh, kw]
        self.ksm_param = nn.Parameter(
            torch.ones(kernel_num, out_channels, in_channels, kd, kh, kw)
        )

        # 5) 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # 6) 智能初始化
        self._init_weights()

    def _init_weights(self):
        """
        基于频谱特性的智能初始化：
        - 低频部分权重更大，高频更小
        - KSM 初始接近 identity（≈1），在训练中微调
        """
        with torch.no_grad():
            for k in range(self.kernel_num):
                scale = 1.0 - (k / self.kernel_num)  # 线性下降: [1.0, ..., ~0]
                self.freq_param.data[:, :, :, :, :, 0] *= scale
                self.freq_param.data[:, :, :, :, :, 1] *= scale

            # KSM 初始为 1 + 很小扰动，避免一开始破坏频域初始化
            self.ksm_param.data.mul_(1.0)
            self.ksm_param.data.add_(torch.randn_like(self.ksm_param) * 1e-3)

    def _get_masks(self, device):
        """
        获取频率掩码（延迟初始化）
        返回: [K,1,1,kd,kh,kw]
        """
        if self.freq_masks is None:
            masks = build_3d_freq_masks(
                self.kd, self.kh, self.kw, self.kernel_num,
                device=device, dtype=torch.bool
            )
            self.freq_masks = masks[:, None, None, ...]
        return self.freq_masks

    def get_kernels(self):
        """
        批量生成所有频带的卷积核 (带 KSM)：
        1) 频域参数 → 应用频带掩码 (FDW)
        2) IFFTN 回到空间域
        3) 与 ksm_param 逐元素相乘 (KSM)
        输出: [K, Cout, Cin, kd, kh, kw]
        """
        device = self.freq_param.device

        masks = self._get_masks(device)  # [K,1,1,kd,kh,kw]
        # 频域参数转复数: [Cout,Cin,kd,kh,kw,2] -> [Cout,Cin,kd,kh,kw]
        freq = torch.view_as_complex(self.freq_param)

        # 扩展成 [K, Cout, Cin, kd,kh,kw] 再乘 mask
        freq = freq.unsqueeze(0)         # [1,Cout,Cin,kd,kh,kw]
        masked_freq = freq * masks       # [K,Cout,Cin,kd,kh,kw]

        # 对最后三个空间维度做 IFFTN
        space_k = fft.ifftn(masked_freq, dim=(-3, -2, -1)).real  # [K,Cout,Cin,kd,kh,kw]

        # KSM: 对每个频带、每个卷积核元素做 element-wise 调制
        space_k = space_k * self.ksm_param  # 逐元素缩放

        return space_k  # [K, Cout, Cin, kd, kh, kw]

    def forward(self, x):
        """
        x: [B, Cin, D, H, W]
        输出: [B, Cout, D_out, H_out, W_out]
        """
        B, C, D, H, W = x.shape
        assert C == self.in_channels, f"输入通道 {C} 不匹配期望 {self.in_channels}"

        # 1) 生成所有频带对应的卷积核（包含 KSM）
        kernels = self.get_kernels()  # [K, Cout, Cin, kd, kh, kw]

        # 2) 生成空间版频带 gate (FBM)
        #    logits: [B,K,D,H,W] -> softmax over K -> gate per voxel
        band_logits = self.band_spatial(x)         # [B, K, D, H, W]
        band_map = self.band_softmax(band_logits)  # [B, K, D, H, W]

        out = 0.0
        band_map_resized = None

        # 3) 对每个频带卷积 + 空间 gate 融合
        for k in range(self.kernel_num):
            w_k = kernels[k]  # [Cout, Cin, kd, kh, kw]
            y_k = F.conv3d(
                x, w_k,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )  # [B, Cout, D_out, H_out, W_out]

            # 首次根据输出大小调整 band_map
            if band_map_resized is None:
                if band_map.shape[2:] == y_k.shape[2:]:
                    band_map_resized = band_map
                else:
                    # stride != 1 等情况，对 gate 用三线性插值对齐输出尺度
                    band_map_resized = F.interpolate(
                        band_map,
                        size=y_k.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )  # [B,K,D_out,H_out,W_out]

            gate_k = band_map_resized[:, k:k+1, ...]  # [B,1,D_out,H_out,W_out]
            y_k = y_k * gate_k                        # 空间版频带 gate

            out = out + y_k

        # 4) 添加偏置
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        return out
