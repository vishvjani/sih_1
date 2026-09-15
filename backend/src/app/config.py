import os

class Config:
    PROJECT_NAME: str = "SignalScope Backend API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DESCRIPTION: str = "Media Authenticity & AI Image Detection API with Unseen Generator Generalization, Calibrated Confidence & Explainable Grad-CAM Heatmaps"

    # Model & Checkpoint defaults
    DEFAULT_BACKBONE: str = "ConvNeXt-Tiny"
    CALIBRATION_TEMPERATURE: float = 1.15
    CHECKPOINT_PATH: str = os.getenv(
        "SIGNALSCOPE_CHECKPOINT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "signalscope_final_calibrated.pth")
    )
    DEVICE: str = os.getenv("SIGNALSCOPE_DEVICE", "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu")
    USE_PYTORCH_MODEL: bool = os.getenv("USE_PYTORCH_MODEL", "1").lower() in ("1", "true", "yes")

config = Config()
