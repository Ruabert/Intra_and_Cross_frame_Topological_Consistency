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
from model.MyModel.blocks import MultiScaleStream_SenP, RegBranch, SDT_to_Edge, SDT_to_Skeleton
from model.MyModel.get_resnet import get_encoder

from model.MyModel.afma import UnetDecoder



logger = logging.getLogger(__name__)

# structure enhanced network
class StructureEnhancedNetworkPlus(nn.Module):
    def __init__(self, arg, config, deepsup=False):
        super(StructureEnhancedNetworkPlus, self).__init__()

        self.config = config
        self.arg_config = arg

        # encoder
        self.encoder = get_encoder(
                "resnet50",
                in_channels=3,
                classes_num=3,
                depth=5,
                weights="imagenet",
                att_depth=1
            )
        # decoder_channels = (256, 128, 64, 32, 16)
        decoder_channels = (1024, 512, 256, 128, 32)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=True if "resnet18".startswith("vgg") else False,
            attention_type=None,
            in_channels=decoder_channels[-1],
            out_classes=3,
            activation=None,
            kernel_size=3,
            att_depth=1
        )

        self.full_seg_head = SegmentationHead(
                             in_channels=32,
                             out_channels=3,
                             kernel_size=1,
                        )


        self.sdt_reg_head = SegmentationHead(
                            in_channels=64,
                            out_channels=2,
                            kernel_size=1,
                        )

        self.dropout = nn.Dropout2d(p=0.1, inplace=False)

        self.MSS = MultiScaleStream_SenP()
        self.graph_node_r = RegBranch()

        # self.ste = SDT_to_Edge()
        # self.skt = SDT_to_Skeleton()
        #
        # self.sdt_act = nn.Tanh()


    def forward(self, x_in):
        # result
        res_dict = {'out_full': None,
                    'out_att': None,
                    'out_sdt': None,
                    'out_ste': None,
                    'out_skt': None}


        # input: bs, frames, 1, h, w
        x_t_1, x_t = x_in[:, 1, ...].repeat(1, 3, 1, 1),  x_in[:, 0, ...].repeat(1, 3, 1, 1)  # x_t: bs, 1, h, w

        bs, c, h, w = x_t.shape

        # t frame extract feature
        features_t, attentions_t = self.encoder(x_t)

        # t_1 frame extract feature
        features_t_1, attentions_t_1 = self.encoder(x_t_1)

        # t frame decode
        dx_t, attentions_t = self.decoder(features_t, attentions_t)

        # t_1 frame decode
        dx_t_1, attentions_t_1 = self.decoder(features_t_1, attentions_t_1)

        # full seg head
        x_t_s = self.full_seg_head(dx_t)
        x_t_1_s = self.full_seg_head(dx_t_1)

        t_seg = self.dropout(x_t_s)
        t_1_seg = self.dropout(x_t_1_s)

        logist_full = torch.cat([t_1_seg.unsqueeze(1), t_seg.unsqueeze(1)], dim=1)
        res_dict['out_full'] = [logist_full]

        logist_att = torch.cat([attentions_t_1.unsqueeze(1), attentions_t.unsqueeze(1)], dim=1)
        res_dict['out_att'] = [logist_att]



        # frames decode the structure

        # features_t = [features_t[-3], features_t[-4]]  # 128, 64
        # ss_in = features_t[-2]  # 256
        depth, mid_hw = x_t.shape[1], x_t.shape[-1]

        x_t_1_o, x_t_2_o, x_t_3_o = self.MSS(features_t[-2], [features_t[-3], features_t[-4]])  # 256, 128, 64
        x_t_1_1_o, x_t_1_2_o, x_t_1_3_o = self.MSS(features_t_1[-2], [features_t_1[-3], features_t_1[-4]])

        # sdt reg head
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

        return res_dict  # out: bs, frames, c, h, w



