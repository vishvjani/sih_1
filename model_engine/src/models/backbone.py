"""
SignalScope ConvNeXt-Tiny Semantic Feature Extractor
Implements transfer learning and differential stage freezing.
"""

import torch
import torch.nn as nn
try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    try:
        import timm
        HAS_TIMM = True
    except ImportError:
        HAS_TIMM = False


class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.out_features = 768

        if HAS_TORCHVISION:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_model = convnext_tiny(weights=weights)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif HAS_TIMM:
            base_model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
            self.features = base_model.stages
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # Standalone fallback feature extractor
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=4, stride=4),
                nn.GELU(),
                nn.Conv2d(96, 192, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(192, 384, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(384, 768, kernel_size=2, stride=2),
                nn.GELU()
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
