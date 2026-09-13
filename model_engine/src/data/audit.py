"""
SignalScope Phase 0: GenImage Dataset Audit & Deduplication Engine
Prevents real-image overlap and format/resolution shortcut leakage.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image


class DatasetAuditor:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    @staticmethod
    def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
        """Computes fast MD5 hash of raw image bytes."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def audit_generator_folder(self, gen_path: Path) -> Dict:
        """
        Audits a single generator folder (e.g. stable_diffusion_v1_4).
        Inspects resolutions, corrupt files, and image counts.
        """
        stats = {
            "name": gen_path.name,
            "real_images": 0,
            "ai_images": 0,
            "corrupt_files": 0,
            "extensions": set(),
            "resolutions": set()
        }

        for split in ["train", "val"]:
            for cls_name, stat_key in [("nature", "real_images"), ("ai", "ai_images")]:
                dir_path = gen_path / split / cls_name
                if not dir_path.exists():
                    continue

                for p in dir_path.glob("*.*"):
                    if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                        continue
                    stats["extensions"].add(p.suffix.lower())
                    try:
                        with Image.open(p) as img:
                            stats["resolutions"].add(img.size)
                            stats[stat_key] += 1
                    except Exception:
                        stats["corrupt_files"] += 1

        stats["extensions"] = list(stats["extensions"])
        stats["resolutions"] = [f"{w}x{h}" for w, h in list(stats["resolutions"])[:5]]
        return stats

    def find_real_image_duplicates(self, gen_paths: List[Path]) -> Tuple[Dict[str, Path], Set[str]]:
        """
        Scans all 'nature' (real) folders across all generator directories.
        Returns:
          unique_real_images: mapping of hash -> canonical image path
          duplicate_hashes: set of hashes that appeared more than once across folders
        """
        seen_hashes = {}
        duplicate_hashes = set()

        for gen_path in gen_paths:
            for split in ["train", "val"]:
                nature_dir = gen_path / split / "nature"
                if not nature_dir.exists():
                    continue
                for img_path in nature_dir.glob("*.*"):
                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                        continue
                    try:
                        h = self.compute_file_hash(img_path)
                        if h in seen_hashes:
                            duplicate_hashes.add(h)
                        else:
                            seen_hashes[h] = img_path
                    except Exception:
                        continue

        return seen_hashes, duplicate_hashes


if __name__ == "__main__":
    print("[SignalScope Phase 0 Audit] Module loaded successfully.")
