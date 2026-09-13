import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Ensure model_engine is on python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.transforms import get_inference_transforms, get_training_transforms
from src.models.network import DualStreamSignalScope
from src.explainability.gradcam import GradCAM
from src.evaluation.metrics import MetricsEvaluator
from src.evaluation.calibration import TemperatureCalibrator


def test_transforms():
    img = Image.new("RGB", (300, 200), color=(120, 80, 40))
    t_train = get_training_transforms(256)
    t_inf = get_inference_transforms(256)

    tensor_train = t_train(img)
    tensor_inf = t_inf(img)

    assert tensor_train.shape == (3, 256, 256), f"Wrong train shape: {tensor_train.shape}"
    assert tensor_inf.shape == (3, 256, 256), f"Wrong inf shape: {tensor_inf.shape}"
    print("[PASS] Transforms test passed.")


def test_network_forward_pass():
    model = DualStreamSignalScope(pretrained=False, use_srm_stream=True)
    model.eval()

    dummy_batch = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_batch)

    assert logits.shape == (2,), f"Expected logits shape (2,), got {logits.shape}"

    features = model.extract_features(dummy_batch)
    assert features["semantic"].shape == (2, 768)
    assert features["forensic"].shape == (2, 256)
    assert features["fused"].shape == (2, 1024)
    print("[PASS] DualStream forward pass and feature extraction passed.")


def test_gradcam():
    model = DualStreamSignalScope(pretrained=False, use_srm_stream=True)
    gradcam = GradCAM(model)
    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)

    heatmap = gradcam.generate_heatmap(dummy_input)
    assert heatmap.shape == (256, 256), f"Heatmap shape mismatch: {heatmap.shape}"
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0, "Heatmap not in [0, 1]"

    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    b64 = gradcam.overlay_heatmap(img, heatmap)
    assert len(b64) > 100, "Invalid base64 string"
    print("[PASS] Grad-CAM heatmap generation passed.")


def test_metrics_and_calibration():
    y_true = [0, 0, 1, 1, 0, 1]
    y_probs = [0.1, 0.2, 0.85, 0.92, 0.4, 0.75]
    gens = ["Real", "Real", "sd_14", "sd_14", "Real", "midjourney"]

    res = MetricsEvaluator.evaluate_predictions(y_true, y_probs, gens, unseen_generator_names=["midjourney"])
    assert "overall_roc_auc" in res
    assert "unseen_generator_roc_auc" in res
    assert res["overall_roc_auc"] > 0.8

    calibrator = TemperatureCalibrator()
    logits = np.array([-2.0, -1.0, 1.5, 2.2, -0.3, 1.1])
    cal_probs = calibrator.calibrate(logits)
    ece = calibrator.compute_ece(cal_probs, np.array(y_true))
    assert 0.0 <= ece <= 1.0
    print("[PASS] Metrics & Calibration test passed.")


if __name__ == "__main__":
    print("Testing SignalScope Core Pipeline...")
    test_transforms()
    test_network_forward_pass()
    test_gradcam()
    test_metrics_and_calibration()
    print("\nALL PIPELINE TESTS PASSED!")
