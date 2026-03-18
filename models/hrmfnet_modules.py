# -*- encoding: utf-8 -*-
'''
@File    :   LPLSNet.py
@Time    :   2026-01-13
@Author  :   niuliang 
@Version :   1.0
@Contact :   niouleung@gmail.com
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, levels=4):
        super().__init__()
        self.levels = levels
        # 使用PyTorch实现的高斯核
        self.gaussian_kernel = self._get_gaussian_kernel()

    def _get_gaussian_kernel(self):
        """创建5x5高斯核"""
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        return kernel.unsqueeze(0).unsqueeze(0)

    def _gaussian_blur(self, x):
        """使用高斯核进行模糊"""
        B, C, H, W = x.shape
        kernel = self.gaussian_kernel.repeat(C, 1, 1, 1).to(x.device)
        return F.conv2d(x, kernel, padding=2, groups=C)

    def _downsample(self, x):
        """下采样"""
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

    def _upsample(self, x, target_size):
        """上采样到目标尺寸"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        """
        构建拉普拉斯金字塔
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            list: 拉普拉斯金字塔层级列表
        """
        # 构建高斯金字塔
        gaussian_pyramid = [x]
        current = x

        for i in range(self.levels):
            blurred = self._gaussian_blur(current)
            downsampled = self._downsample(blurred)
            gaussian_pyramid.append(downsampled)
            current = downsampled

        # 构建拉普拉斯金字塔
        laplacian_pyramid = []
        for i in range(self.levels):
            current_level = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]

            # 上采样下一层到当前层尺寸
            upsampled = self._upsample(next_level, current_level.shape[2:])

            # 计算拉普拉斯层
            laplacian = current_level - upsampled
            laplacian_pyramid.append(laplacian)

        # 添加最后一层（最小的高斯层）
        laplacian_pyramid.append(gaussian_pyramid[-1])

        # 返回高斯金字塔用于替代AvgPool（低通 + 抗混叠）
        return gaussian_pyramid


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DCU(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.pw2(self.relu(self.pw1(self.dw(x)))))
        return x * attn


class MFIE(nn.Module):
    def __init__(self, channels, num_branches=4, dilations=(1, 2, 3, 4)):
        super().__init__()
        if num_branches != 4:
            raise ValueError("MFIE expects num_branches=4 for dilations (1,2,3,4).")
        if len(dilations) != 4:
            raise ValueError("MFIE expects 4 dilation values for 4 branches.")
        self.num_branches = 4
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        base = channels // num_branches
        split_sizes = [base] * num_branches
        split_sizes[-1] += channels - base * num_branches
        self.split_sizes = split_sizes

        self.branches = nn.ModuleList(
            [DCU(split_sizes[i], dilation=dilations[i]) for i in range(num_branches)]
        )
        self.fuse = DCU(channels, dilation=1)

    def _channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        if c % groups != 0:
            return x
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        x_in = self.pre(x)
        splits = torch.split(x_in, self.split_sizes, dim=1)
        outs = [branch(s) for branch, s in zip(self.branches, splits)]
        x_cat = torch.cat(outs[:3], dim=1)
        x_shuf = self._channel_shuffle(x_cat, 3)
        x_mix = torch.cat([x_shuf, outs[3]], dim=1)
        fused = self.fuse(x_mix)
        return x_in + fused


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, cshf_kernel=3):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.cshf_kernel = cshf_kernel
        self.cshf_padding = cshf_kernel // 2

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # CSHF: min/max pool -> add -> sigmoid -> gate skip
        min_pool = -F.max_pool2d(-x1, kernel_size=self.cshf_kernel, stride=1, padding=self.cshf_padding)
        max_pool = F.max_pool2d(x1, kernel_size=self.cshf_kernel, stride=1, padding=self.cshf_padding)
        # attn = torch.sigmoid(max_pool + min_pool) # 0.8297
        attn = torch.sigmoid(max_pool - min_pool)

        x2_gated = x2 * attn

        x = torch.cat([x2_gated, x1], dim=1)
        output = self.conv(x)
        
        # 返回CSHF之后的特征图和最终输出
        return output, x2_gated


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
