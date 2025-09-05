import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd, _pair




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

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, deepsup=True):
        super().__init__()
        self.config = config
        self.deepsup = deepsup
        head_channels = 1024
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, hidden_states, features=None):
        deepsup_list = []
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.view(B, hidden, h, w).contiguous()
        # x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            deepsup_list.append(x)


        hh, ww = deepsup_list[-1].shape[-2], deepsup_list[-1].shape[-1]
        for idx, f in enumerate(deepsup_list[:-1]):
            deepsup_list[idx] = F.interpolate(f, (hh, ww), mode='bilinear', align_corners=True)

        if self.deepsup:
            return deepsup_list
        else:
            return [deepsup_list[-1]]

class EdgeFocusedAtt(nn.Module):
    def __init__(self, f_channels, e_channels, emb_channels=128):
        super(EdgeFocusedAtt, self).__init__()

        self.f_channels = f_channels
        self.e_channels = e_channels
        self.emb_channels = emb_channels


        self.k_conv = nn.Conv2d(in_channels=self.e_channels, out_channels=self.emb_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)

        self.q_conv = nn.Conv2d(in_channels=self.f_channels, out_channels=self.emb_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)

        self.v_conv = nn.Conv2d(in_channels=self.f_channels, out_channels=self.f_channels,
                                kernel_size=1, stride=1, padding=0, bias=False)

        self.attn_dropout = nn.Dropout()

    def forward(self, f_in, e_in):
        B = f_in.shape[0]

        k = self.k_conv(e_in)
        k = k.view(B * self.emb_channels, -1).contiguous()  # B, C, H, W -> B, C, H * W -> B, H * W, C

        q = self.q_conv(f_in)
        q = q.view(B * self.emb_channels, -1).contiguous()

        v = self.v_conv(f_in)
        v = v.view(B * self.emb_channels, -1).contiguous()

        similarity_map = torch.matmul(q, k.T)
        similarity_map = F.softmax(similarity_map, dim=-1)
        similarity_map = self.attn_dropout(similarity_map)

        out = torch.matmul(similarity_map, v)
        out = out.view(B, self.e_channels, *f_in.shape[2:]).contiguous()
        return f_in + out

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, norm_layer=nn.BatchNorm2d, bias=False):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self.in_LN = norm_layer(in_channels + 1)
        self.out_LN = norm_layer(1)
        self.out_act = nn.ReLU()

        self._gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        f_in = torch.cat([input_features, gating_features], dim=1)

        # f_in = f_in.contiguous().transpose(1, -1)
        # f_in = f_in.permute(0, 2, 3, 1).contiguous()

        f_in = self.in_LN(f_in)
        # f_in = f_in.contiguous().transpose(-1, 1)
        # f_in = f_in.permute(0, 3, 1, 2).contiguous()

        alphas = self._gate_conv(f_in)
        # alphas = alphas.contiguous().transpose(1, -1)
        # alphas = alphas.permute(0, 2, 3, 1).contiguous()

        alphas = self.out_LN(alphas)
        # alphas = alphas.contiguous().transpose(-1, 1)
        # alphas = alphas.permute(0, 3, 1, 2).contiguous()

        alphas = self.out_act(alphas)
        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)



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

        self.rfb1 = MSRF(256, 64)
        self.rfb2 = MSRF(512, 64)
        self.rfb3 = MSRF(1024, 64)


    def forward(self, f_in, features):
        # Shape Stream: Focus the edge feature of our cnn feature map
        # features: 512, 256
        x1_o = self.rfb1(features[1])  # 256 to 64
        x2_o = self.rfb2(features[0])  # 512 to 64
        x3_o = self.rfb3(f_in)  # 1024 to 64

        return x1_o, x2_o, x3_o

class MultiScaleStream_SenP(nn.Module):
    def __init__(self):
        super(MultiScaleStream_SenP, self).__init__()

        self.rfb1 = MSRF(256, 64)
        self.rfb2 = MSRF(512, 64)
        self.rfb3 = MSRF(1024, 64)


    def forward(self, f_in, features):

        # Shape Stream: Focus the edge feature of our cnn feature map
        # features: 512, 256

        x3_o = self.rfb3(f_in)  # 256 to 64

        x1_o = self.rfb1(features[1])  # 64 to 64

        x2_o = self.rfb2(features[0])  # 128 to 64


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

class SDT_to_Edge(nn.Module):
    def __init__(self, k=100):
        super(SDT_to_Edge, self).__init__()

        self.k = k
        self.neg_inhibit_act = nn.SiLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):

        x = self.neg_inhibit_act(x)
        x = (1 - self.out_act(self.k * x))

        return x

class SDT_to_Skeleton(nn.Module):
    def __init__(self, t=0.3):
        super(SDT_to_Skeleton, self).__init__()

        self.t = t
        self.out_act = nn.ReLU()


    def forward(self, x):
        x = self.out_act(x - self.t)
        return x