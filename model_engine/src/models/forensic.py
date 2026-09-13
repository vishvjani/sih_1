"""
SignalScope Spatial Rich Model (SRM) Forensic High-Pass Stream
Extracts high-frequency noise residuals R = I - smooth(I) to expose latent diffusion & GAN checkerboards.
"""

import numpy as np
import torch
import torch.nn as nn


class SRMHighPassFilter(nn.Module):
    """
    Applies fixed, non-trainable Spatial Rich Model (SRM) high-pass kernels.
    Extracts 3 classic steganalysis / forensic residual filters across RGB channels.
    """
    def __init__(self):
        super().__init__()
        # Filter 1: 1st order edge high-pass (5x5)
        f1 = np.array([
            [0,  0,  0,  0,  0],
            [0, -1,  2, -1,  0],
            [0,  2, -4,  2,  0],
            [0, -1,  2, -1,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0

        # Filter 2: 2nd order Laplacian (5x5)
        f2 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        # Filter 3: 3rd order square high-pass (5x5)
        f3 = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  1,  0,  0],
            [0,  1, -4,  1,  0],
            [0,  0,  1,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0

        # 3 filters per RGB channel = 9 output feature maps
        weights = np.zeros((9, 3, 5, 5), dtype=np.float32)
        for i, f in enumerate([f1, f2, f3]):
            for c in range(3):
                weights[i * 3 + c, c, :, :] = f

        self.srm_conv = nn.Conv2d(3, 9, kernel_size=5, stride=1, padding=2, bias=False)
        self.srm_conv.weight = nn.Parameter(torch.from_numpy(weights), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.srm_conv(x)


class SRMForensicStream(nn.Module):
    """
    Lightweight forensic convolutional encoder that converts SRM noise residuals into a 256-d feature vector.
    """
    def __init__(self, out_features: int = 256):
        super().__init__()
        self.srm = SRMHighPassFilter()
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.srm(x)
        return self.encoder(residuals)
