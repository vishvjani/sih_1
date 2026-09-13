"""
SignalScope PyTorch Dataset Loader
Loads manifest-indexed samples with metadata for generator-aware evaluation.
"""

import json
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset
import torch


class GenImageDataset(Dataset):
    def __init__(self, manifest_path: str, transform: Optional[Callable] = None):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        img_path = record["path"]
        label = float(record["label"])
        generator = record.get("generator", "Unknown")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Robust fallback for corrupted read
            image = Image.new("RGB", (256, 256), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            import numpy as np
            arr = np.array(image, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std
            image = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": generator,
            "path": str(img_path)
        }
