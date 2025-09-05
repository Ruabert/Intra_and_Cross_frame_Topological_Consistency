import os
import argparse
import yaml

def get_parser():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation')

    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')

    parser.add_argument('--model', type=str, default='', help='model name')

    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')

    parser.add_argument('--consistency', type=float, default=0.1, help='consistency_weight_max')

    parser.add_argument('--consistency_rampup', type=int, default=100, help='consistency_rampup_epoch')

    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--warm_epoch', type=int, default=0)

    parser.add_argument('--base_lr', default=1e-3)  # normal 5e-4 fineture 2e-5

    parser.add_argument('--lr_min', default=1e-5)  # normal 1e-5 fineture 1e-6

    parser.add_argument('--lr_warm', default=1e-6)

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')

    parser.add_argument('--load', type=str, default=False, help='load pretrained model')

    parser.add_argument('--cp_path', type=str, default='', help='checkpoint path')

    parser.add_argument('--log_path', type=str, default='../tf-logs/', help='log path')

    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')
    # Ablation Stud
    parser.add_argument('--ab_name', type=str, default=None)
    
    parser.add_argument('--kk_thresh', type=float, default=0.8)
    
    parser.add_argument('--labeled_rate', type=float, default=0.1)
    
    parser.add_argument('--consistency_weight_1', type=float, default=0.01)
    
    parser.add_argument('--consistency_weight_2', type=float, default=0.01)
    
    parser.add_argument('--debug_ratio', type=int, default=1)


    args = parser.parse_args()
    
    if args.ab_name == None: 
        config_path = 'config/%s/%s_%s.yaml' % (args.dataset, args.model, args.dimension)
    else:
        config_path = 'config/%s/%s_%s_%s.yaml' % (args.dataset, args.model, args.dimension, args.ab_name)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args