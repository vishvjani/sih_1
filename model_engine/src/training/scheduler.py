"""
SignalScope Discriminative Learning Rate & Parameter Scheduler
Protects pretrained representations with differential layer-wise learning rates.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    phase: int = 1,
    phase1_lr: float = 0.001,
    phase2_backbone_lr: float = 0.00005,
    phase2_head_lr: float = 0.0002,
    weight_decay: float = 0.01,
    total_epochs: int = 7
):
    if phase == 1:
        # Train classifier only
        params = [p for p in model.classifier.parameters() if p.requires_grad]
        if hasattr(model, "forensic_stream"):
            params += [p for p in model.forensic_stream.encoder.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=phase1_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=1e-5)
    else:
        # Phase 2: Discriminative LR
        backbone_params = []
        for param in model.semantic_stream.features.parameters():
            if param.requires_grad:
                backbone_params.append(param)

        head_params = list(model.classifier.parameters())
        if hasattr(model, "forensic_stream"):
            head_params += list(model.forensic_stream.encoder.parameters())

        param_groups = [
            {"params": backbone_params, "lr": phase2_backbone_lr},
            {"params": head_params, "lr": phase2_head_lr}
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

        if total_epochs > 1:
            # Warmup (1 epoch) + Cosine Decay
            warmup = LinearLR(optimizer, start_factor=0.2, total_iters=1)
            cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - 1, 1), eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)

    return optimizer, scheduler
