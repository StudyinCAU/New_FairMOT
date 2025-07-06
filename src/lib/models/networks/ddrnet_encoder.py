from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmseg.models.backbones.ddrnet import DDRNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ddrnet_encoder(nn.Module):
    def __init__(self, variant_cfg, heads, head_conv=256):
        super(ddrnet_encoder, self).__init__()
        self.heads = heads

        self.backbone = DDRNet(
            in_channels=variant_cfg.get('in_channels', 3),
            channels=variant_cfg.get('channels', 32),
            ppm_channels=variant_cfg.get('ppm_channels', 128),
            align_corners=variant_cfg.get('align_corners', False),
            norm_cfg=variant_cfg.get('norm_cfg', dict(type='BN', requires_grad=True)),
            act_cfg=variant_cfg.get('act_cfg', dict(type='ReLU', inplace=True)),
        )

        # DDRNet输出为 x_s + x_c，通道数为 channels * 4
        out_channels = variant_cfg.get('channels', 32) * 4

        self.deconv = nn.Sequential(
            nn.Conv2d(out_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        feat = self.backbone(x)  # 输出: Tensor (batch_size, c, h, w)
        feat = self.deconv(feat)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return [ret]


def get_pose_net(num_layers, heads, head_conv=256):
    # ddrnet23
    if num_layers == 23:
        variant_cfg = dict(
            in_channels=3,
            channels=32,
            ppm_channels=128,
            align_corners=False,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True)
        )
    else:
        raise ValueError(f"Unsupported DDRNet variant num_layers={num_layers}")

    model = ddrnet_encoder(variant_cfg, heads, head_conv)
    return model
