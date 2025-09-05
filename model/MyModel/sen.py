# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from model.MyModel.blocks import *
from model.MyModel import model_configs as configs
from model.MyModel.resnet import ResNetV2
from model.MyModel.blocks import MultiScaleStream, RegBranch, SDT_to_Edge, SDT_to_Skeleton

logger = logging.getLogger(__name__)

# structure enhanced network
class StructureEnhancedNetwork(nn.Module):
    def __init__(self, arg, config, deepsup=False):
        super(StructureEnhancedNetwork, self).__init__()

        self.config = config
        self.arg_config = arg

        # encoder
        self.encoder = ResNetV2(block_units=config.resnet.num_layers, width_factor=1)

        # self.transformer = SetTimeSformer(img_size=224, in_chans=self.encoder.width * 16, patch_size=16,
        #                                     num_classes=401408, num_frames=2)  #  num_classes = 49152

        self.decoder = DecoderCup(config, deepsup=False)



        self.full_seg_head = []
        for dc in config['decoder_channels']:
            self.full_seg_head.append(
                SegmentationHead(
                    in_channels=dc,
                    out_channels=3,
                    kernel_size=1,
                ).cuda()
            )


        self.sdt_reg_head = SegmentationHead(
                            in_channels=64,
                            out_channels=2,
                            kernel_size=1,
                        )

        self.dropout = nn.Dropout2d(p=0.1, inplace=False)

        self.mss = MultiScaleStream()
        self.graph_node_r = RegBranch()

        self.ste = SDT_to_Edge()
        self.skt = SDT_to_Skeleton()

        self.sdt_act = nn.Tanh()


    def forward(self, x_in):
        # result
        res_dict = {'out_full': None,
                    'out_sdt': None,
                    'out_ste': None,
                    'out_skt': None}


        # input: bs, frames, 1, h, w
        x_t_1, x_t = x_in[:, 1, ...].repeat(1, 3, 1, 1),  x_in[:, 0, ...].repeat(1, 3, 1, 1)  # x_t: bs, 1, h, w

        bs, c, h, w = x_t.shape

        x_t, features_t = self.encoder(x_t)
        x_t_1, features_t_1 = self.encoder(x_t_1)

        depth, mid_hw = x_t.shape[1], x_t.shape[-1]

        # x = torch.cat([x_t_1.unsqueeze(1), x_t.unsqueeze(1)], dim=1).transpose(1, 2)  # t,c -> c,t

        # x = self.transformer(x)

        # x = x.contiguous().view(bs, 2, mid_hw * mid_hw, depth)
        # x = x.contiguous().view(bs, 2, depth, mid_hw, mid_hw)  # bs, 2, depth, mid_hw, mid_hw

        # x_t_1, x_t = x[:, 0, ...], x[:, -1, ...]

        x_t_1_o, x_t_2_o, x_t_3_o = self.mss(x_t, features_t[:-1])  # 512, 256
        x_t_1_1_o, x_t_1_2_o, x_t_1_3_o = self.mss(x_t_1, features_t_1[:-1])

        x_t_r = self.graph_node_r(x_t_3_o, x_t_2_o, x_t_1_o)
        x_t_r = self.sdt_reg_head(x_t_r)
        t_reg = self.dropout(x_t_r)
        x_t_1_r = self.graph_node_r(x_t_1_3_o, x_t_1_2_o, x_t_1_1_o)
        x_t_1_r = self.sdt_reg_head(x_t_1_r)
        t_1_reg = self.dropout(x_t_1_r)

        logist_sdt = torch.cat([t_1_reg.unsqueeze(1), t_reg.unsqueeze(1)], dim=1)

        res_dict['out_sdt'] = [logist_sdt]  # out_sdt: bs, frames, c, h, w

        # # get sdt map to edge
        # t_1_edge = self.ste(self.sdt_act(t_1_reg))
        # t_edge = self.ste(self.sdt_act(t_reg))
        #
        # # get sdt map to skt
        # t_1_skt = self.skt(self.sdt_act(t_1_reg))
        # t_skt = self.skt(self.sdt_act(t_reg))
        #
        # logist_edge = torch.cat([t_1_edge.unsqueeze(1), t_edge.unsqueeze(1)], dim=1)
        # logist_skeleton = torch.cat([t_1_skt.unsqueeze(1), t_skt.unsqueeze(1)], dim=1)
        #
        # res_dict['out_ste'] = [logist_edge]
        # res_dict['out_skt'] = [logist_skeleton]


        x_t_1, x_t = x_t_1.contiguous().view(bs, mid_hw * mid_hw, depth), x_t.contiguous().view(bs, mid_hw * mid_hw, depth)  # bs, mid_hw * mid_hw, depth

        dx_t = self.decoder(x_t, features_t)
        dx_t_1 = self.decoder(x_t_1, features_t_1)  # bs, num_class, h, w

        x_t_s = self.full_seg_head[-1](dx_t[-1])
        x_t_1_s = self.full_seg_head[-1](dx_t_1[-1])

        t_seg = self.dropout(x_t_s)
        t_1_seg = self.dropout(x_t_1_s)

        logist_full = torch.cat([t_1_seg.unsqueeze(1), t_seg.unsqueeze(1)], dim=1)
        res_dict['out_full'] = [logist_full]


        return res_dict  # out: bs, frames, c, h, w

CONFIGS = {'R50-ViT-B_16': configs.get_r50_b16_config()}

if __name__ == '__main__':
    sen = StructureEnhancedNetwork(CONFIGS['R50-ViT-B_16'])

    x = torch.randn(1, 3, 64, 64)
    res_dict = sen(x)


    print(res_dict)

