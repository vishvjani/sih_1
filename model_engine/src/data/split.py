"""
SignalScope Generator-Aware Split Partition Engine
Enforces zero data leakage and isolates unseen generators for evaluation.
"""

import json
import random
from pathlib import Path
from typing import Dict, List


class GeneratorSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def create_100k_manifests(
        self,
        unique_real_paths: List[Path],
        ai_paths_by_generator: Dict[str, List[Path]],
        output_dir: Path
    ) -> Dict[str, str]:
        """
        Partitions 100,000 samples:
          Train (70,000): 35,000 Real + 35,000 AI (Seen: SD1.4, GLIDE, Wukong, BigGAN)
          Val (10,000):    5,000 Real +  5,000 AI (SD1.5 + held-out seen classes)
          Test (20,000):  10,000 Real + 10,000 AI (Unseen: Midjourney, VQDM)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        random.shuffle(unique_real_paths)

        # 1. Allocate 50,000 Real images with zero overlap
        real_train = unique_real_paths[:35000]
        real_val = unique_real_paths[35000:40000]
        real_test = unique_real_paths[40000:50000]

        # 2. Allocate AI Generators
        # Train AI (35k): SD 1.4 (12k), GLIDE (8k), Wukong (8k), BigGAN (7k)
        ai_train = []
        train_targets = {
            "stable_diffusion_v1_4": 12000,
            "glide": 8000,
            "wukong": 8000,
            "biggan": 7000
        }
        for gen_name, target in train_targets.items():
            paths = ai_paths_by_generator.get(gen_name, [])
            random.shuffle(paths)
            ai_train.extend([(str(p), gen_name) for p in paths[:target]])

        # Val AI (5k): SD 1.5 (3k) + remaining seen (2k)
        ai_val = []
        sd15_paths = ai_paths_by_generator.get("stable_diffusion_v1_5", [])
        random.shuffle(sd15_paths)
        ai_val.extend([(str(p), "stable_diffusion_v1_5") for p in sd15_paths[:3000]])
        # Add 2k from remaining seen pools
        for gen_name, target in train_targets.items():
            paths = ai_paths_by_generator.get(gen_name, [])
            rem = paths[target:target + 500]
            ai_val.extend([(str(p), gen_name) for p in rem])

        # Test AI (10k UNSEEN): Midjourney (6k), VQDM (4k)
        ai_test = []
        mj_paths = ai_paths_by_generator.get("midjourney", [])
        vqdm_paths = ai_paths_by_generator.get("vqdm", [])
        random.shuffle(mj_paths)
        random.shuffle(vqdm_paths)
        ai_test.extend([(str(p), "midjourney (UNSEEN)") for p in mj_paths[:6000]])
        ai_test.extend([(str(p), "vqdm (UNSEEN)") for p in vqdm_paths[:4000]])

        # 3. Assemble full records
        def build_records(real_list, ai_list):
            records = []
            for p in real_list:
                records.append({"path": str(p), "label": 0, "generator": "ImageNet (Real)", "is_ai": False})
            for p, gen in ai_list:
                records.append({"path": p, "label": 1, "generator": gen, "is_ai": True})
            random.shuffle(records)
            return records

        train_records = build_records(real_train, ai_train)
        val_records = build_records(real_val, ai_val)
        test_records = build_records(real_test, ai_test)

        manifests = {}
        for name, data in [("train", train_records), ("val", val_records), ("test_unseen", test_records)]:
            file_path = output_dir / f"{name}_manifest.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            manifests[name] = str(file_path)

        return manifests
