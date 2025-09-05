import numpy as np
import torch
import torch.nn as nn
import pdb

BASELINEMODEL = ['unet', 'resunet', 'transunet', 'swinunet', 'convformer', 'unext', 'utnet', 'medformer']
OURMODEL = ['USEN', 'USEN_UFLOW']



def get_model(args, pretrain=False):
    # set model mode
    if args.model in BASELINEMODEL:
        args.model_mode = 'baseline'
    elif args.model in OURMODEL:
        args.model_mode = 'our'

    # baseline
    if args.model == 'unet':

        from .dim2 import UNet
        return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)

    elif args.model == 'resunet':
        from .dim2 import UNet
        return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)

    elif args.model == 'medformer':
        from .dim2 import MedFormer
        return MedFormer(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.ReLU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, aux_loss=args.aux_loss)

    elif args.model == 'transunet':
        from .dim2 import VisionTransformer as ViT_seg
        from .dim2.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(args.training_size[0]/16), int(args.training_size[1]/16))
        net = ViT_seg(config_vit, img_size=args.training_size[0], num_classes=args.classes)
        return net

    elif args.model == 'swinunet':
        from .dim2 import SwinUnet
        from .dim2.swin_unet import SwinUnet_config
        config = SwinUnet_config()
        net = SwinUnet(config, img_size=64, num_classes=args.classes)
        return net

    elif args.model == 'convformer':
        from .dim2 import ConvFormer
        return ConvFormer(args.in_chan, args.classes, args.training_size[0])

    elif args.model == 'setr':
        from .dim2 import Setr
        return Setr(args.in_chan, args.classes, args.training_size[0])

    elif args.model == 'unext':
        from .dim2 import UNext
        return UNext(args.classes, args.in_chan, args.training_size[0])

    elif args.model == 'utnet':
        from .dim2 import UTNet
        return UTNet(in_chan=args.in_chan, base_chan=32, num_classes=args.classes)

    elif args.model == 'USEN':
        from .MyModel import UStructureEnhancedNetwork
        net = UStructureEnhancedNetwork(args)
        return net

    elif args.model == 'USEN_UFLOW':
        from .MyModel import UStructureEnhancedNetwork_UFlow
        net = UStructureEnhancedNetwork_UFlow(args)
        return net


