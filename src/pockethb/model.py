"""Hb regressor model: ImageNet-pretrained backbone + linear regression head."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

_BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
}


class HbRegressor(nn.Module):
    """Backbone (frozen except top stages by default) → linear head → scalar Hb in g/dL."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone: {backbone}")
        ctor, weights, head_attr = _BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)

        if head_attr == "fc":
            self.feature_dim = net.fc.in_features
            net.fc = nn.Identity()
        else:
            # efficientnet: classifier is (Dropout, Linear). steal the Linear's in_features.
            self.feature_dim = net.classifier[-1].in_features
            net.classifier = nn.Identity()

        self.backbone_name = backbone
        self.backbone = net
        self.head = nn.Linear(self.feature_dim, 1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, 3, H, W) — already ImageNet-normalised
        feat = self.backbone(x)
        return self.head(feat).squeeze(-1)
