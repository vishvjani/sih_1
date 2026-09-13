"""
SignalScope Dual-Stream Architecture: ConvNeXt-Tiny (Semantic) + SRM (Forensic)
Fuses scene semantics with high-frequency noise residuals for generalization to unseen generators.
"""

import torch
import torch.nn as nn
from .backbone import ConvNeXtTinyBackbone
from .forensic import SRMForensicStream


class DualStreamSignalScope(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_srm_stream: bool = True
    ):
        super().__init__()
        self.use_srm_stream = use_srm_stream
        self.semantic_stream = ConvNeXtTinyBackbone(pretrained=pretrained)
        in_dim = self.semantic_stream.out_features  # 768

        if self.use_srm_stream:
            self.forensic_stream = SRMForensicStream(out_features=256)
            in_dim += 256  # 768 + 256 = 1024

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logit (1-d per sample). Apply sigmoid for raw probability.
        """
        sem_feat = self.semantic_stream(x)

        if self.use_srm_stream:
            for_feat = self.forensic_stream(x)
            fused_feat = torch.cat([sem_feat, for_feat], dim=1)
        else:
            fused_feat = sem_feat

        logits = self.classifier(fused_feat).squeeze(-1)
        return logits

    def extract_features(self, x: torch.Tensor) -> dict:
        sem_feat = self.semantic_stream(x)
        for_feat = self.forensic_stream(x) if self.use_srm_stream else None
        fused = torch.cat([sem_feat, for_feat], dim=1) if for_feat is not None else sem_feat
        return {
            "semantic": sem_feat,
            "forensic": for_feat,
            "fused": fused
        }

    def freeze_backbone(self):
        """Phase 1: Train classifier head only."""
        self.semantic_stream.freeze_all()

    def unfreeze_later_stages(self):
        """Phase 2: Fine-tune Stage 3 & 4 of ConvNeXt."""
        self.semantic_stream.unfreeze_later_stages()
