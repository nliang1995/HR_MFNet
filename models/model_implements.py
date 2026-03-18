# -*- encoding: utf-8 -*-
'''
@File    :   model_implements.py
@Time    :   2025-12-15
@Author  :   niuliang 
@Version :   1.0
@Contact :   niouleung@gmail.com
'''


import torch.nn as nn
import torch
import torch.nn.functional as F

from collections import OrderedDict

from models.hrmfnet_modules import *


class HR_MFNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, bilinear=True, **kwargs):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.laplacian_pyramid = LaplacianPyramid(levels=4)   
        
        self.inc = DoubleConv(in_channels, 64)
        self.mfie1 = MFIE(64)
        self.down1 = Down(64, 128)
        self.mfie2 = MFIE(128)
        self.down2 = Down(128, 256)
        self.mfie3 = MFIE(256)
        self.down3 = Down(256, 512)
        self.mfie4 = MFIE(512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.mfie5 = MFIE(1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 融合卷积：将拼接后的通道数映射回原阶段通道数
        # 对应 x2(128c), x3(256c), x4(512c), x5(1024//factor c)
        self.fuse2 = nn.Conv2d(128 + in_channels, 128, kernel_size=1, bias=False)
        self.fuse3 = nn.Conv2d(256 + in_channels, 256, kernel_size=1, bias=False)
        self.fuse4 = nn.Conv2d(512 + in_channels, 512, kernel_size=1, bias=False)
        self.fuse5 = nn.Conv2d(1024 // factor + in_channels, 1024 // factor, kernel_size=1, bias=False)
        
        # 用于保存CSHF特征图
        self.cshf_features = {}

    def forward(self, x):
        
        pyramid_levels = self.laplacian_pyramid(x)
        x_down = pyramid_levels[1:5]

        x1 = self.inc(x)
        x1 = self.mfie1(x1)
        x2 = self.down1(x1)
        # 将 H/2 尺度的金字塔图 concat 到 x2 并用 1x1 融合回 128 通道
        x2 = torch.cat([x2, x_down[0]], dim=1)
        x2 = self.fuse2(x2)
        x2 = self.mfie2(x2)

        x3 = self.down2(x2)
        x3 = torch.cat([x3, x_down[1]], dim=1)
        x3 = self.fuse3(x3)
        x3 = self.mfie3(x3)

        x4 = self.down3(x3)
        x4 = torch.cat([x4, x_down[2]], dim=1)
        x4 = self.fuse4(x4)
        x4 = self.mfie4(x4)

        x5 = self.down4(x4)
        x5 = torch.cat([x5, x_down[3]], dim=1)
        x5 = self.fuse5(x5)
        x5 = self.mfie5(x5)

        x, cshf1 = self.up1(x5, x4)
        self.cshf_features['up1'] = cshf1.detach().cpu().numpy()
        x, cshf2 = self.up2(x, x3)
        self.cshf_features['up2'] = cshf2.detach().cpu().numpy()
        x, cshf3 = self.up3(x, x2)
        self.cshf_features['up3'] = cshf3.detach().cpu().numpy()
        x, cshf4 = self.up4(x, x1)
        self.cshf_features['up4'] = cshf4.detach().cpu().numpy()
        logits = self.outc(x)

        return torch.sigmoid(logits)
    
