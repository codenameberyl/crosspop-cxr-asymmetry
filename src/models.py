"""
Backbone architectures and instantiation utilities.

Constructs ImageNet-pretrained models (MobileNetV2, EfficientNet-B0) adapted for binary 
classification with matching head architecture patterns.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_model(arch, pretrained=True, num_classes=2):
    """Build classification model based on selected backbone network architecture."""
    if arch == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported architecture specified: {arch!r}")

    return model


def count_parameters(model):
    """Return dictionary containing total and trainable model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
