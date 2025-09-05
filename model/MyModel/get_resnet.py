from copy import deepcopy
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.MyModel.afma import Encoder_channelatt_img
new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 64, 64],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


encoders_channelatt_img = {
    "resnet18": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet50": {
        "encoder": Encoder_channelatt_img,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
}


encoders = {}
encoders.update(encoders_channelatt_img)

def get_encoder(name, in_channels=3, classes_num=12, depth=5, att_depth=99, weights=None, replace_stride_with_dilation=None):
    try:
        Encoder = encoders[name]["encoder"]

    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported utils: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(att_depth=att_depth)
    params.update(replace_stride_with_dilation=replace_stride_with_dilation)
    params.update(classes_num=classes_num)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                weights, name, list(encoders[name]["pretrained_settings"].keys()),
            ))
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder

if __name__ == '__main__':
    encoder = get_encoder(
        "resnet18",
        in_channels=3,
        depth=5,
        weights="imagenet",
        att_depth=1
    )


    x = torch.randn(1, 3, 64, 64)
    features, attentions = encoder(x)


    print(encoder)