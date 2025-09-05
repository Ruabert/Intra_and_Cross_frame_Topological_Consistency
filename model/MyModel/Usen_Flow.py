import cv2
import torch
import torch.nn as nn
from model.MyModel.utils import get_block
from model.MyModel.unet_utils import inconv, down_block, up_block
from dataset.dim2.utils import to_skeleton_distance_transform
import numpy as np

from model.MyModel.uflow import UFlow



class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            dilation=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not (use_batchnorm),
        )
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class MSRF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSRF, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            Conv2dReLU(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            Conv2dReLU(in_channel, out_channel, 1),
            Conv2dReLU(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            Conv2dReLU(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            Conv2dReLU(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            Conv2dReLU(in_channel, out_channel, 1),
            Conv2dReLU(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            Conv2dReLU(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            Conv2dReLU(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            Conv2dReLU(in_channel, out_channel, 1),
            Conv2dReLU(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            Conv2dReLU(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            Conv2dReLU(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = Conv2dReLU(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = Conv2dReLU(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class MultiScaleStream(nn.Module):
    def __init__(self):
        super(MultiScaleStream, self).__init__()

        self.rfb1 = MSRF(128, 64)
        self.rfb2 = MSRF(256, 64)
        self.rfb3 = MSRF(512, 64)


    def forward(self, f_in, features):  # 512, 256, 128
        # Shape Stream: Focus the edge feature of our cnn feature map
        # features: 512, 256
        x1_o = self.rfb1(features[1])  # 128 to 64
        x2_o = self.rfb2(features[0])  # 256 to 64
        x3_o = self.rfb3(f_in)  # 512 to 64

        return x1_o, x2_o, x3_o

class RegBranch(nn.Module):
    def __init__(self, channel=64, out_channel=2):
        super(RegBranch, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = Conv2dReLU(channel, channel, 3, padding=1)
        self.conv_upsample2 = Conv2dReLU(channel, channel, 3, padding=1)
        self.conv_upsample3 = Conv2dReLU(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = Conv2dReLU(channel, channel, 3, padding=1)
        self.conv_2 = Conv2dReLU(channel, channel, 3, padding=1)
        self.conv_3 = Conv2dReLU(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = Conv2dReLU(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = Conv2dReLU(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = Conv2dReLU(4 * channel, 4 * channel, 3, padding=1)

        self.de_conv1 = Conv2dReLU(4 * channel, 2 * channel, 3, padding=1)
        self.de_conv2 = Conv2dReLU(channel * 2, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))

        x = self.de_conv1(self.upsample(cat_x4))
        x = self.de_conv2(self.upsample(x))

        return x

class UNetMSRF(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='BasicBlock', pool=True):
        super().__init__()

        block = get_block(block)
        nb = 2  # num_block

        # Unet
        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2 * base_ch, num_block=nb, block=block, pool=pool)
        self.down2 = down_block(2 * base_ch, 4 * base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4 * base_ch, 8 * base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8 * base_ch, 16 * base_ch, num_block=nb, block=block, pool=pool)

        self.up1 = up_block(16 * base_ch, 8 * base_ch, num_block=nb, block=block)
        self.up2 = up_block(8 * base_ch, 4 * base_ch, num_block=nb, block=block)
        self.up3 = up_block(4 * base_ch, 2 * base_ch, num_block=nb, block=block)
        self.up4 = up_block(2 * base_ch, base_ch, num_block=nb, block=block)

        # SDT decoder
        self.MSS = MultiScaleStream()
        self.graph_node_r = RegBranch()

        # segmentation head
        self.full_seg_head = SegmentationHead(
            in_channels=32,
            out_channels=3,
            kernel_size=1,
        )

        # regression head
        self.sdt_reg_head = SegmentationHead(
            in_channels=64,
            out_channels=2,
            kernel_size=1,
        )

        self.dropout = nn.Dropout2d(p=0.1, inplace=False)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # mask decoder
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        # segmentation
        out_seg = self.full_seg_head(out)

        # sdt decoder
        x_1_o, x_2_o, x_3_o = self.MSS(x5, [x4, x3])
        x_r = self.graph_node_r(x_3_o, x_2_o, x_1_o)

        # regression
        x_r = self.sdt_reg_head(x_r)
        out_reg = self.dropout(x_r)

        out_reg_no_head = x_r
        out_seg_no_head = out

        return out_seg, out_reg, out_seg_no_head



class UStructureEnhancedNetwork_UFlow(nn.Module):
    def __init__(self, arg=None):
        super(UStructureEnhancedNetwork_UFlow, self).__init__()

        self.arg_config = arg

  
        self.feature_extractor = UNetMSRF(in_ch=1,
                                 num_classes=3,
                                 base_ch=32,
                                 block='BasicBlock', pool=True)


    
        # transfer mask pred to edge
        # self.mask_to_edge = Mask_to_Edge()

        # uflow
        self.optical_flow_extractor = UFlow(num_channels=1,
                                            num_levels=2,
                                            channel_multiplier=2,
                                            use_cost_volume=True,
                                            action_channels=0)

        
        self.sdt_act = nn.Tanh()
        self.mask_act = nn.Sigmoid()

    def forward(self, x_in):
        # result
        res_dict = {'out_full': None,
                    'out_sdt': None,
                    'pseudo_mask': None,
                    'optical_flow': [],
                    'feature_pyramid_1': [],
                    'feature_pyramid_2': []}


        # input: bs, frames, 1, h, w
        x_t_1, x_t = x_in[:, 0, ...],  x_in[:, 1, ...]  # x_t: bs, 1, h, w

        bs, c, h, w = x_t.shape

        # t frame into pipeline
        t_seg,  t_reg, _ = self.feature_extractor(x_t)

        # t-1 frame into pipeline
        t_1_seg, t_1_reg, _ = self.feature_extractor(x_t_1)

        # get segmentation map, for supervised learning
        logist_full = torch.cat([t_1_seg.unsqueeze(1), t_seg.unsqueeze(1)], dim=1)
        res_dict['out_full'] = [logist_full]

        # get sdt map, for supervised learning
        logist_sdt = torch.cat([t_1_reg.unsqueeze(1), t_reg.unsqueeze(1)], dim=1)
        res_dict['out_sdt'] = [logist_sdt]  # out_sdt: bs, frames, c-1, h, w

        # get sdt map to mask, for unsupervised learning
        t_1_rm = self.sdt_act(t_1_reg)
        t_rm = self.sdt_act(t_reg)

        # add bg
        bg_mask = torch.zeros_like(t_rm[:, 0, ...]).unsqueeze(1)
        
        cls_mask_1 = torch.ones_like(t_rm[:, 0, ...]).unsqueeze(1)
        cls_mask_2 = torch.ones_like(t_rm[:, 0, ...]).unsqueeze(1)
        cls_mask = torch.cat([cls_mask_1, cls_mask_2], dim=1)
        
        
        t_1_rm = torch.cat([bg_mask, t_1_rm], dim=1)
        t_rm = torch.cat([bg_mask, t_rm], dim=1)

        pseudo_mask = torch.cat([t_1_rm.unsqueeze(1), t_rm.unsqueeze(1)], dim=1)
        res_dict['pseudo_mask'] = [pseudo_mask]

        # input to the Uflow
        c = t_1_reg.shape[1]
        flow_list = []
        fp_1 = []
        fp_2 = []
        
        
        for k in range(c):
            flow, feature_pyramid_1, feature_pyramid_2 = self.optical_flow_extractor(t_1_reg[:, k, ...].unsqueeze(1), t_reg[:, k, ...].unsqueeze(1))

            flow_list.append(flow)
            fp_1.append(feature_pyramid_1)
            fp_2.append(feature_pyramid_2)

            
        # flow, feature_pyramid_1, feature_pyramid_2 = self.optical_flow_extractor(t_1_se, t_se)
        

        res_dict['optical_flow'] = flow_list
        res_dict['feature_pyramid_1'] = fp_1
        res_dict['feature_pyramid_2'] = fp_2


        return res_dict  # out: bs, frames, c, h, w





if __name__ == '__main__':
    x = torch.randn(32, 2, 1, 64, 64)
    net = UStructureEnhancedNetwork_UFlow()
    out = net(x)

    # 网络训练过程：（1）前100轮半监督训练usen；（2）冻结usen，固定输出的edge，无监督训练uflow；（3）冻结uflow，借光度一致损失继续学习usen

    print(out.shape)