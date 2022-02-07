import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import vit_pytorch as vp


def get_model(name: str, height: int, width: int, in_channels: int, num_classes: int, pretrained: bool = False, **kwargs):
    if name == "mobilevit":
        return None

    elif name == "vit_for_small_dataset":
        return None

    else:
        return timm.create_model(name, pretrained=pretrained)
