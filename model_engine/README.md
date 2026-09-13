# SignalScope Deep Learning Research & Training Engine

SignalScope is an advanced Computer Vision forensic engine for media authenticity detection, built specifically to achieve **generalization to unseen AI generators** without shortcut learning or data leakage.

---

## 🏗 Modular Architecture

```text
model_engine/
├── configs/
│   └── training_config.yaml         # Configurable hyperparameters & paths
├── src/
│   ├── data/
│   │   ├── audit.py                 # Phase 0: GenImage audit & hash deduplication
│   │   ├── dataset.py               # Generator-aware PyTorch Dataset
│   │   └── split.py                 # Leakage-free generator partitioner (Train vs Unseen Test)
│   ├── preprocessing/
│   │   └── transforms.py            # Anti-shortcut resize, crop, & augmentations
│   ├── models/
│   │   ├── backbone.py              # ConvNeXt-Tiny semantic feature extractor (768-d)
│   │   ├── forensic.py              # Spatial Rich Model (SRM) high-pass noise stream (256-d)
│   │   └── network.py               # Dual-Stream fusion architecture (1024-d)
│   ├── training/
│   │   ├── trainer.py               # Mixed-precision (AMP) two-phase trainer
│   │   └── scheduler.py             # Discriminative learning rate scheduler
│   ├── evaluation/
│   │   ├── metrics.py               # Overall ROC-AUC, Unseen-Generator ROC-AUC, Macro-F1, FPR
│   │   └── calibration.py           # Platt Temperature Scaling & Expected Calibration Error (ECE)
│   ├── explainability/
│   │   └── gradcam.py               # Real PyTorch Grad-CAM visual attribution
│   └── inference/
│       └── predict.py               # Unified prediction & explainability contract
└── tests/
    └── test_pipeline.py             # Pipeline verification tests
```

---

## 🔬 Scientific Methodology

### 1. Zero-Leakage 100,000 Image Partitioning
- **Real Image De-duplication**: Real ImageNet images across generator folders are hashed using MD5/pHash to ensure zero overlap between train and test.
- **Seen vs Unseen Split**:
  - **Train (70,000: 35k Real, 35k AI)**: Stable Diffusion v1.4, GLIDE, Wukong, BigGAN.
  - **Val (10,000: 5k Real, 5k AI)**: Stable Diffusion v1.5 + held-out seen classes.
  - **Unseen Test Benchmark (20,000: 10k Real, 10k AI)**: **Midjourney + VQDM** (Zero exposure during training/validation).

### 2. Dual-Stream ConvNeXt-Tiny + SRM Residuals
- **Semantic Stream**: Pretrained ConvNeXt-Tiny extracts 768-d high-level semantic features.
- **Forensic Stream**: 3 fixed Spatial Rich Model (SRM) filters subtract low-frequency scene semantics and extract 256-d high-frequency noise residuals $R = I - \text{smooth}(I)$.
- **Feature Fusion**: 1024-d fused representation passed to a calibrated binary classification head.

### 3. Two-Phase Differential Training
- **Phase 1**: Backbone frozen, classification head warmup ($10^{-3}$ LR).
- **Phase 2**: Stage 3 & 4 fine-tuning with discriminative learning rates ($\eta_{\text{backbone}}=5\times 10^{-5}$, $\eta_{\text{head}}=2\times 10^{-4}$) and Cosine Annealing.

---

## 🚀 Running on Google Colab (Free T4 GPU)

1. Open **`colab/SignalScope_Master_Colab_Trainer.ipynb`** in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells sequentially.
4. The trained checkpoint (`signalscope_final_calibrated.pth`) and calibration config are exported directly to your Google Drive folder (`SignalScope_Checkpoints/`).
