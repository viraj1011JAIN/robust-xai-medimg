from __future__ import annotations

import timm

from .hooks import FeatureExtractor


def build_model(arch: str = "resnet50", num_classes: int = 1, pretrained: bool = True):
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    if arch.startswith("resnet"):
        target = model.layer4
    elif arch.startswith("efficientnet_b0"):
        target = model.conv_head
    elif arch in (
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_patch16_224.augreg_in21k",
    ):
        target = model.blocks[-1].norm1
    else:
        target = list(model.children())[-1]
    hook = FeatureExtractor(target)
    return model, hook
