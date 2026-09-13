"""
SignalScope Confidence Calibration Engine (Platt / Temperature Scaling)
Minimizes Expected Calibration Error (ECE) to eliminate overconfident predictions.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, lr: float = 0.01, max_iter: int = 100):
        """Optimizes scalar temperature T on validation logits to minimize NLL."""
        if val_logits is None or len(val_logits) == 0:
            self.temperature = 1.0
            return self.temperature

        try:
            logits_t = torch.tensor(val_logits, dtype=torch.float32)
            labels_t = torch.tensor(val_labels, dtype=torch.float32)

            t_param = nn.Parameter(torch.ones(1) * 1.5)
            optimizer = optim.LBFGS([t_param], lr=lr, max_iter=max_iter)
            criterion = nn.BCEWithLogitsLoss()

            def eval_loss():
                optimizer.zero_grad()
                scaled_logits = logits_t / torch.clamp(t_param, min=0.1)
                loss = criterion(scaled_logits, labels_t)
                loss.backward()
                return loss

            optimizer.step(eval_loss)
            self.temperature = float(torch.clamp(t_param, min=0.1).item())
        except Exception:
            self.temperature = 1.0
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error across n_bins."""
        bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(probs)

        for i in range(n_bins):
            bin_lower = bin_limits[i]
            bin_upper = bin_limits[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(labels[in_bin])
                conf_in_bin = np.mean(probs[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

        return float(ece)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"calibrated_temperature": self.temperature}, f, indent=2)
