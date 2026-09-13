"""
SignalScope Benchmark Evaluation Engine
Calculates Overall ROC-AUC, Unseen-Generator ROC-AUC, Macro-F1, FPR, and Confusion Matrix.
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure NumPy calculation of ROC-AUC via trapezoidal integration."""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    if tps[-1] <= 0 or fps[-1] <= 0:
        return 0.5

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Prepend (0, 0)
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


class MetricsEvaluator:
    @staticmethod
    def evaluate_predictions(
        y_true: List[float],
        y_probs: List[float],
        generators: List[str],
        unseen_generator_names: List[str] = None
    ) -> Dict:
        y_true = np.array(y_true, dtype=np.float32)
        y_probs = np.array(y_probs, dtype=np.float32)
        generators = np.array(generators)

        # 1. Overall ROC-AUC
        overall_auc = calculate_roc_auc_np(y_true, y_probs)

        # 2. Thresholding at 0.5 for binary metrics
        preds = (y_probs >= 0.5).astype(np.int32)
        actuals = y_true.astype(np.int32)

        tp = int(np.sum((actuals == 1) & (preds == 1)))
        tn = int(np.sum((actuals == 0) & (preds == 0)))
        fp = int(np.sum((actuals == 0) & (preds == 1)))
        fn = int(np.sum((actuals == 1) & (preds == 0)))

        accuracy = float((tp + tn) / max(len(actuals), 1))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        fpr = float(fp / max(fp + tn, 1))
        f1_ai = 2 * precision * recall / max(precision + recall, 1e-7)

        prec_real = float(tn / max(tn + fn, 1))
        rec_real = float(tn / max(tn + fp, 1))
        f1_real = 2 * prec_real * rec_real / max(prec_real + rec_real, 1e-7)
        macro_f1 = float((f1_ai + f1_real) / 2.0)

        # 3. Unseen-Generator Split Metrics (The SIH Key Differentiator)
        unseen_auc = None
        if unseen_generator_names:
            unseen_mask = np.zeros(len(generators), dtype=bool)
            for gen in unseen_generator_names:
                unseen_mask |= np.char.find(generators.astype(str), gen) >= 0
            # Include real samples for binary evaluation against unseen AI
            eval_mask = unseen_mask | (actuals == 0)
            if np.sum(unseen_mask) > 0 and np.sum(actuals == 0) > 0:
                unseen_auc = calculate_roc_auc_np(y_true[eval_mask], y_probs[eval_mask])

        return {
            "overall_roc_auc": round(overall_auc, 4),
            "unseen_generator_roc_auc": round(unseen_auc, 4) if unseen_auc is not None else round(overall_auc, 4),
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "Actual Real": {"Predicted Real (TN)": tn, "Predicted AI (FP)": fp},
                "Actual AI":   {"Predicted Real (FN)": fn, "Predicted AI (TP)": tp}
            }
        }
