"""
SignalScope ConvNeXt-Tiny Semantic Feature Extractor
Implements transfer learning and differential stage freezing.
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base_model = convnext_tiny(weights=weights)

        # Retain feature extractor (features block contains all 4 stages)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def freeze_all(self):
        """Freezes entire backbone for Phase 1 classifier warmup."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_later_stages(self):
        """
        Unfreezes later stages (Stage 3 & 4) for Phase 2 fine-tuning.
        Early texture/edge stages (0, 1, 2) remain frozen to preserve general vision representations.
        """
        # ConvNeXt features has 8 layers: 0=stem, 1=stage0, 2=downsample, 3=stage1, 4=downsample, 5=stage2, 6=downsample, 7=stage3
        for i, layer in enumerate(self.features):
            if i >= 5:  # Stage 2 and Stage 3 + downsamplers
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
