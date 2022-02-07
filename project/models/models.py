import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from vit_pytorch import vit_for_small_dataset


def get_model(name: str, height: int, width: int, in_channels: int, num_classes: int, pretrained: bool = False, **kwargs):
    if name == "mobilevit":
        return None

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
