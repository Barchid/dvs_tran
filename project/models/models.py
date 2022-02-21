import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from vit_pytorch import vit_for_small_dataset, mobile_vit
import torchvision.models as models


def get_resnet18(in_channels: int, num_classes: int):
    resnet18 = models.resnet18(progress=True)

    resnet18.fc = nn.Linear(512, num_classes, bias=True)

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return resnet18


def get_model(name: str, height: int, width: int, in_channels: int, num_classes: int, pretrained: bool = False, **kwargs):
    if name == "mobilevit_xxs":
        model = mobile_vit.MobileViT(
            image_size=(height, width),
            dims=[64, 80, 96],
            channels=[16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
            num_classes=num_classes
        )
        model.conv1 = mobile_vit.conv_nxn_bn(in_channels, 16, stride=2)
        return model

    elif name == "mobilevit_xs":
        model = mobile_vit.MobileViT(
            image_size=(height, width),
            dims=[96, 120, 144],
            channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
            num_classes=num_classes
        )
        model.conv1 = mobile_vit.conv_nxn_bn(in_channels, 16, stride=2)
        return model

    elif name == "mobilevit_s":
        model = mobile_vit.MobileViT(
            image_size=(height, width),
            dims=[144, 192, 240],
            channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
            num_classes=num_classes
        )
        model.conv1 = mobile_vit.conv_nxn_bn(in_channels, 16, stride=2)
        return model

    elif name == "vit_for_small_dataset":
        return vit_for_small_dataset.ViT(
            image_size=(height, width),
            patch_size=16,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    else:
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels)
        model.reset_classifier(num_classes)

        return model
