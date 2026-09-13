"""
SignalScope Mixed-Precision Two-Phase Trainer
Executes Phase 1 (warmup) and Phase 2 (fine-tuning) with automatic checkpointing and unseen-generator logging.
"""

import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .scheduler import build_optimizer_and_scheduler
from ..evaluation.metrics import MetricsEvaluator
from ..evaluation.calibration import TemperatureCalibrator


class SignalScopeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_unseen_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        unseen_generator_names: list = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_unseen_loader = test_unseen_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.unseen_generator_names = unseen_generator_names or ["midjourney", "vqdm"]

        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        self.best_unseen_auc = 0.0
        self.best_val_auc = 0.0

    def train_epoch(self, optimizer, scaler) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple_Eval:
        self.model.eval()
        all_logits = []
        all_labels = []
        all_generators = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            all_logits.extend(logits.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_generators.extend(batch["generator"])

        probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
        metrics = MetricsEvaluator.evaluate_predictions(
            y_true=all_labels,
            y_probs=probs.tolist(),
            generators=all_generators,
            unseen_generator_names=self.unseen_generator_names
        )
        metrics["loss"] = round(total_loss / max(num_batches, 1), 4)
        return metrics, np.array(all_logits), np.array(all_labels)

    def train(
        self,
        phase1_epochs: int = 3,
        phase2_epochs: int = 7,
        phase1_lr: float = 1e-3,
        phase2_backbone_lr: float = 5e-5,
        phase2_head_lr: float = 2e-4
    ) -> Dict:
        print("\n=======================================================")
        print("🚀 SignalScope Dual-Stream Training Pipeline Initiated")
        print(f"Device: {self.device} | Mixed Precision: {self.device == 'cuda'}")
        print("=======================================================\n")

        # ----------------- PHASE 1: WARMUP HEAD -----------------
        print(f"[Phase 1] Training classification head ({phase1_epochs} epochs, backbone FROZEN)...")
        self.model.freeze_backbone()
        opt1, sched1 = build_optimizer_and_scheduler(self.model, phase=1, phase1_lr=phase1_lr, total_epochs=phase1_epochs)

        for epoch in range(1, phase1_epochs + 1):
            t0 = time.time()
            loss = self.train_epoch(opt1, self.scaler)
            sched1.step()
            val_metrics, _, _ = self.evaluate(self.val_loader)
            el = round(time.time() - t0, 1)

            print(f"  Epoch [{epoch}/{phase1_epochs}] ({el}s) - Train Loss: {loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val ROC-AUC: {val_metrics['overall_roc_auc']:.4f}")

        # ----------------- PHASE 2: DIFFERENTIAL FINE-TUNING -----------------
        print(f"\n[Phase 2] Differential Fine-Tuning Stage 3 & 4 ({phase2_epochs} epochs)...")
        self.model.unfreeze_later_stages()
        opt2, sched2 = build_optimizer_and_scheduler(
            self.model, phase=2, phase2_backbone_lr=phase2_backbone_lr, phase2_head_lr=phase2_head_lr, total_epochs=phase2_epochs
        )

        for epoch in range(1, phase2_epochs + 1):
            t0 = time.time()
            loss = self.train_epoch(opt2, self.scaler)
            sched2.step()
            val_metrics, val_logits, val_labels = self.evaluate(self.val_loader)

            # Evaluate on held-out unseen generator benchmark
            unseen_auc = 0.0
            if self.test_unseen_loader:
                test_metrics, _, _ = self.evaluate(self.test_unseen_loader)
                unseen_auc = test_metrics["unseen_generator_roc_auc"]

            el = round(time.time() - t0, 1)
            print(
                f"  Epoch [{epoch}/{phase2_epochs}] ({el}s) - Loss: {loss:.4f} | Val AUC: {val_metrics['overall_roc_auc']:.4f} | "
                f"Unseen-Gen AUC: {unseen_auc:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            # Checkpoint best unseen AUC
            if unseen_auc > self.best_unseen_auc:
                self.best_unseen_auc = unseen_auc
                torch.save(self.model.state_dict(), self.checkpoint_dir / "signalscope_best_unseen_auc.pth")
                print(f"    ⭐ New Best Unseen-Gen ROC-AUC: {unseen_auc:.4f} -> Checkpoint saved!")

            if val_metrics["overall_roc_auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["overall_roc_auc"]
                torch.save(self.model.state_dict(), self.checkpoint_dir / "signalscope_best_val_auc.pth")

        # ----------------- PHASE 3: CALIBRATION -----------------
        print("\n[Phase 3] Optimizing Platt Temperature Scaling on validation logits...")
        calibrator = TemperatureCalibrator()
        t_val = calibrator.fit(val_logits, val_labels)
        calibrator.save(str(self.checkpoint_dir / "calibration_config.json"))
        print(f"  Optimized Temperature T = {t_val:.4f} (Saved to calibration_config.json)")

        # Save final complete production weights
        final_path = self.checkpoint_dir / "signalscope_final_calibrated.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "calibrated_temperature": t_val,
            "best_unseen_auc": self.best_unseen_auc,
            "best_val_auc": self.best_val_auc
        }, final_path)
        print(f"\n✅ Production checkpoint saved: {final_path}")

        return {
            "best_unseen_auc": self.best_unseen_auc,
            "best_val_auc": self.best_val_auc,
            "calibrated_temperature": t_val
        }
