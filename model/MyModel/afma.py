# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from copy import deepcopy
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

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
from typing import Optional, Union


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class Encoder_channelatt_img(ResNet, EncoderMixin):
    def __init__(self, out_channels, classes_num=12, patch_size=8, depth=5, att_depth=1, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._attention_on_depth=att_depth

        self._out_channels = out_channels

        self._in_channels = 3

        self.patch_size = patch_size

        self.conv_img=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7),padding=3),

            nn.Conv2d(64, classes_num, kernel_size=(3,3), padding=1)
        )

        self.conv_feamap=nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], classes_num, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.resolution_trans=nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2*self.patch_size * self.patch_size, bias=False),
            nn.Linear(2*self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        attentions=[]

        x = stages[0](x)
        features.append(x)

        ini_img=self.conv_img(x)

        x = stages[1](x)
        features.append(x)

        if self._attention_on_depth == 1:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):

                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att=torch.unsqueeze(att,1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        x = stages[2](x)
        features.append(x)

        if self._attention_on_depth == 2:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att=torch.unsqueeze(att,1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)


        x = stages[3](x)
        features.append(x)

        if self._attention_on_depth == 3:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[4](x)
        features.append(x)

        if self._attention_on_depth == 4:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):

                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[5](x)
        features.append(x)

        if self._attention_on_depth == 5:
            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        return features, attentions


    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        #state_dict.pop("conv_and_unfold_list")

        super().load_state_dict(state_dict, strict=False, **kwargs)

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        # elif name == 'scse':
        #     self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            in_channels=777,
            out_classes=777,
            activation: Optional[Union[str, callable]] = None,
            kernel_size=3,
            att_depth=-1,
    ):
        super().__init__()

        seg_input_channels=in_channels
        seg_output_channels=out_classes

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        # self.segmentation = SegmentationHead(in_channels=seg_input_channels, out_channels=seg_output_channels,
        #                                      kernel_size=kernel_size, activation=activation, att_depth=att_depth)

    def forward(self, features, attentions):
        features = features[1:]
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        # x, attentions = self.segmentation(x, attentions)

        return x, attentions