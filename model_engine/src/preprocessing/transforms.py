"""
SignalScope Anti-Shortcut Preprocessing & Robust Augmentations
Prevents resolution, compression, and aspect-ratio shortcut learning.
"""

import io
import random
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SimulatedJPEGCompression:
    """Simulates real-world social media JPEG recompression."""
    def __init__(self, quality_min: int = 65, quality_max: int = 95, p: float = 0.4):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def get_training_transforms(target_size: int = 256):
    """
    Training pipeline with forensic-preserving robust augmentations.
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        SimulatedJPEGCompression(quality_min=65, quality_max=95, p=0.4),
        T.RandomResizedCrop(target_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_inference_transforms(target_size: int = 256):
    """
    Deterministic inference and validation pipeline.
    Aspect-ratio preserving resize followed by center crop.
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
