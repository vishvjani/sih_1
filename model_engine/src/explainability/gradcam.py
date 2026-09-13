"""
SignalScope PyTorch Grad-CAM Explainability Engine
Computes visual attribution heatmaps highlighting localized synthetic artifact regions.
"""

import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model.eval()
        # Default target layer is the final stage block of ConvNeXt
        if target_layer is None:
            self.target_layer = self.model.semantic_stream.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        input_tensor: [1, 3, H, W]
        Returns: 2D numpy array [H, W] normalized in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Target AI logit
        output.backward(gradient=torch.ones_like(output))

        # Channel-wise global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input image resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> str:
        """
        Overlays heatmap on original PIL image and returns base64 PNG string.
        """
        w, h = image.size
        heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.Resampling.BILINEAR)
        heatmap_np = np.array(heatmap_resized, dtype=np.float32) / 255.0

        # Simple colormap (Red for high intensity, Blue for low)
        r = np.uint8(np.clip(2.0 * heatmap_np, 0, 1) * 255)
        g = np.uint8(np.clip(2.0 * (1.0 - np.abs(heatmap_np - 0.5)), 0, 1) * 200)
        b = np.uint8(np.clip(2.0 * (1.0 - heatmap_np), 0, 1) * 255)
        color_heatmap = np.stack([r, g, b], axis=-1)

        img_np = np.array(image.convert("RGB"))
        blended = np.uint8(img_np * (1.0 - alpha) + color_heatmap * alpha)
        result_img = Image.fromarray(blended)

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
