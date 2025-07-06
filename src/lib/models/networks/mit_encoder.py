from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmseg.models.backbones.mit import MixVisionTransformer

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

embed_dims_map = {
    'b0': [32, 64, 160, 256],
    'b1': [64, 128, 320, 512],
    'b2': [64, 128, 320, 512],
    'b3': [64, 128, 320, 512],
    'b4': [64, 128, 320, 512],
    'b5': [64, 128, 320, 512],
}


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class mit_encoder(nn.Module):
    def __init__(self, variant, heads, head_conv=256):
        super(mit_encoder, self).__init__()
        self.heads = heads

        # 使用 MixVisionTransformer backbone
        self.backbone = MixVisionTransformer(
            in_channels=3,
            embed_dims=embed_dims_map[variant][0],
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            pretrained=None  # 使用 mmseg 加载权重，或者先不用
        )
        self.variant = variant

        sum_out_dim = sum(embed_dims_map[variant])
        # 输出 heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(sum_out_dim, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(sum_out_dim, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        feat = self.backbone(x)
        x0_h, x0_w = feat[0].size(2), feat[0].size(3)
        x1 = F.upsample(feat[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(feat[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(feat[3], size=(x0_h, x0_w), mode='bilinear')
        feat = torch.cat([feat[0], x1, x2, x3], 1)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return [ret]


def get_pose_net(num_layers, heads, head_conv=256):
    variant = {0: 'b0', 1: 'b1', 2: 'b2', 3: 'b3', 4: 'b4', 5: 'b5'}.get(num_layers, 'b2')
    model = mit_encoder(variant, heads, head_conv)
    return model
