from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.backbones.vit import VisionTransformer

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class vit_encoder(nn.Module):
    def __init__(self, variant, heads, head_conv=256):
        super(vit_encoder, self).__init__()
        self.heads = heads
        embed_dims = 768
        # 定义 ViT 主干
        self.backbone = VisionTransformer(
            img_size=(608, 1088),  # 输入尺寸 (h, w)
            patch_size=16,
            in_channels=3,
            embed_dims=embed_dims,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            out_indices=(2, 5, 8, 11),
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            with_cls_token=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            interpolate_mode='bicubic'
        )

        # heads 输出层
        sum_out_dim = embed_dims * 4  # ViT 输出的特征维度
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
                fc = nn.Conv2d(256, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        feat = self.backbone(x)  # 输出为 list
        x0_h, x0_w = feat[0].size(2), feat[0].size(3)   # （batch_size, c, h, w）
        x1 = F.upsample(feat[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(feat[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(feat[3], size=(x0_h, x0_w), mode='bilinear')
        feat = torch.cat([feat[0], x1, x2, x3], 1)
        feat = self.deconv(feat)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        return [ret]


def get_pose_net(num_layers, heads, head_conv=256):
    variant = 'vit_b16'
    model = vit_encoder(variant, heads, head_conv)
    return model
