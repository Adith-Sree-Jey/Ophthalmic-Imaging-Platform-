"""
gradcam_engine.py — Grad-CAM heatmap generator for the V7 multimodal cataract model.

Hooks into the anterior-segment backbone's last conv layer and produces:
  1. A colourised heatmap overlaid on the original image (saved as PNG)
  2. A structured region description dict for MedGemma to reason about

Place this file in the backend application directory.

Usage (called internally by classifier.py / report pipeline):
    from gradcam_engine import GradCAMAnalyser
    analyser = GradCAMAnalyser(model, device)
    heatmap_rgb, description = analyser.run(ant_tensor, ant_rgb_np, pred_class_idx)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


# ── Region vocabulary ─────────────────────────────────────────────────────────
# The anterior image is divided into a 3x3 grid.
# Each cell is labelled anatomically for the clinical description.

_REGION_LABELS = [
    ["superior-nasal",   "superior",   "superior-temporal"],
    ["nasal",            "central",    "temporal"],
    ["inferior-nasal",   "inferior",   "inferior-temporal"],
]


class GradCAMAnalyser:
    """
    Runs Grad-CAM on the anterior-segment branch of the V7 model.

    Parameters
    ----------
    model : MultiModalCataractModelV7
        The already-loaded V7 model (eval mode).
    device : torch.device
        CUDA or CPU.
    target_layer_name : str
        Dot-path to the last conv layer inside the anterior backbone.
        Default works for EfficientNet/ConvNeXt backbones used in V7.
        Override if your backbone differs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        target_layer_name: str = "ant_encoder.features",
        modality: str = "anterior",
    ):
        self.model  = model
        self.device = device
        self.modality = modality

        # Resolve the target layer
        self._layer = self._resolve_layer(target_layer_name)
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []

    # ── Layer resolution ──────────────────────────────────────────────────────

    def _resolve_layer(self, layer_name: str) -> torch.nn.Module:
        """
        Walk the dot-path to find the layer.
        Falls back gracefully if the name doesn't match exactly.
        """
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Try to find the last Sequential / Conv2d we can hook
                module = self._find_last_conv(self.model)
                print(f"[GradCAM] Layer '{layer_name}' not found — "
                      f"using auto-detected layer: {type(module).__name__}")
                return module
        # If we landed on a container (Sequential etc.), get its last child
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            module = list(module.children())[-1]
        return module

    @staticmethod
    def _find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
        """Walk the model and return the last Conv2d layer found."""
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("GradCAM: no Conv2d layer found in model.")
        return last_conv

    # ── Hook management ───────────────────────────────────────────────────────

    def _register_hooks(self):
        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks = [
            self._layer.register_forward_hook(_save_activation),
            self._layer.register_full_backward_hook(_save_gradient),
        ]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ── Grad-CAM core ─────────────────────────────────────────────────────────

    def _compute_cam(
        self,
        ant_tensor: torch.Tensor,
        rg_tensor: torch.Tensor,
        sl_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Forward + backward pass to compute the raw CAM for class_idx.
        Returns a 2-D float32 array in [0, 1].
        """
        self.model.zero_grad()
        self._register_hooks()

        try:
            # torch.enable_grad() lets gradients flow even when model is in eval()
            # This avoids the BatchNorm "expected more than 1 value" crash that
            # occurs when .train() is used with batch_size=1.
            with torch.enable_grad():
                ant_t = ant_tensor.to(self.device).requires_grad_(True)
                rg_t  = rg_tensor.to(self.device)
                sl_t  = sl_tensor.to(self.device)

                # Use only the CE head for Grad-CAM (most direct classification signal)
                ce_logits, _, _, _ = self.model(ant_t, rg_t, sl_t, return_attention=True)

                # Backward on the predicted class score
                score = ce_logits[0, class_idx]
                score.backward()

            # Grad-CAM formula: global-average-pool gradients → weight activations
            gradients   = self._gradients   # (1, C, H, W)
            activations = self._activations # (1, C, H, W)

            if gradients is None or activations is None:
                return np.zeros((7, 7), dtype=np.float32)

            weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam     = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)
            cam     = F.relu(cam)

            # Normalise to [0, 1]
            cam_np = cam.cpu().numpy()
            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()
            return cam_np.astype(np.float32)

        finally:
            self._remove_hooks()
            self.model.zero_grad()

    # ── Overlay generation ────────────────────────────────────────────────────

    @staticmethod
    def _overlay_heatmap(
        cam: np.ndarray,
        image_rgb: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Resize CAM to image size, apply JET colormap, blend with original."""
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = image_rgb.astype(np.float32).copy()
        blended = (
            (1 - alpha) * image_rgb.astype(np.float32)
            + alpha * heatmap_rgb.astype(np.float32)
        ).clip(0, 255)
        if roi_mask is None:
            overlay = blended
        else:
            roi = roi_mask.astype(bool)
            overlay[roi] = blended[roi]
        overlay = overlay.astype(np.uint8)
        return overlay

    @staticmethod
    def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected contour from a binary mask."""
        mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask_u8)
        largest = max(contours, key=cv2.contourArea)
        kept = np.zeros_like(mask_u8)
        cv2.drawContours(kept, [largest], -1, 255, -1)
        return kept

    @staticmethod
    def _center_ellipse_mask(shape: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask,
            center=(w // 2, h // 2),
            axes=(max(1, int(w * scale_x)), max(1, int(h * scale_y))),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )
        return mask

    def _build_anterior_roi_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        return self._center_ellipse_mask((h, w), 0.36, 0.36)

    def _build_red_glow_roi_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 80], dtype=np.uint8)
        upper_red1 = np.array([25, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([155, 50, 80], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        colour_mask = cv2.bitwise_or(mask1, mask2)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(colour_mask, bright_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = self._largest_component(combined)
        if combined.sum() == 0:
            return self._center_ellipse_mask(gray.shape, 0.42, 0.42)

        ys, xs = np.where(combined > 0)
        cx = int(xs.mean())
        cy = int(ys.mean())
        radius = int(max(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()), min(gray.shape) * 0.28))
        mask = np.zeros_like(combined)
        cv2.circle(mask, (cx, cy), max(1, int(radius * 1.05)), 255, -1)
        return mask

    def _build_slit_lamp_roi_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        bright = self._largest_component(bright)
        if bright.sum() == 0:
            return self._center_ellipse_mask(gray.shape, 0.38, 0.44)

        ys, xs = np.where(bright > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad_x = max(4, int((x2 - x1) * 0.12))
        pad_y = max(4, int((y2 - y1) * 0.12))
        mask = np.zeros_like(bright)
        cv2.rectangle(
            mask,
            (max(0, x1 - pad_x), max(0, y1 - pad_y)),
            (min(gray.shape[1] - 1, x2 + pad_x), min(gray.shape[0] - 1, y2 + pad_y)),
            255,
            -1,
        )
        return mask

    def _build_roi_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.modality == "red_glow":
            return self._build_red_glow_roi_mask(image_rgb)
        if self.modality == "slit_lamp":
            return self._build_slit_lamp_roi_mask(image_rgb)
        return self._build_anterior_roi_mask(image_rgb)

    # ── Region analysis ───────────────────────────────────────────────────────

    @staticmethod
    def _analyse_regions(cam: np.ndarray) -> dict:
        """
        Divide the CAM into a 3x3 grid and compute activation statistics
        for each anatomical region. Returns a structured description dict.
        """
        h, w = cam.shape
        thirds_h = [0, h // 3, 2 * h // 3, h]
        thirds_w = [0, w // 3, 2 * w // 3, w]

        region_scores: dict[str, float] = {}
        for row in range(3):
            for col in range(3):
                patch = cam[
                    thirds_h[row]:thirds_h[row + 1],
                    thirds_w[col]:thirds_w[col + 1],
                ]
                label = _REGION_LABELS[row][col]
                region_scores[label] = float(patch.mean())

        # Sort by activation strength
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
        top3    = [r for r, _ in sorted_regions[:3]]
        bottom3 = [r for r, _ in sorted_regions[-3:]]

        total = sum(region_scores.values()) + 1e-8
        central_fraction = region_scores.get("central", 0) / total

        # Intensity characterisation
        max_val = cam.max()
        if max_val > 0.75:
            intensity = "high"
        elif max_val > 0.40:
            intensity = "moderate"
        else:
            intensity = "low"

        # Spatial spread
        active_pixels   = (cam > 0.3).sum()
        total_pixels    = cam.size
        spread_fraction = active_pixels / total_pixels
        if spread_fraction > 0.5:
            spread = "diffuse"
        elif spread_fraction > 0.25:
            spread = "regional"
        else:
            spread = "focal"

        return {
            "top_regions":        top3,
            "low_regions":        bottom3,
            "central_fraction":   round(central_fraction, 3),
            "activation_intensity": intensity,
            "activation_spread":    spread,
            "region_scores":      {k: round(v, 4) for k, v in region_scores.items()},
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        ant_tensor: torch.Tensor,
        rg_tensor: torch.Tensor,
        sl_tensor: torch.Tensor,
        image_rgb: np.ndarray,
        pred_class_idx: int,
    ) -> tuple[np.ndarray, dict]:
        """
        Run Grad-CAM and return the overlay image + region analysis.

        Parameters
        ----------
        ant_tensor : torch.Tensor  shape (1, 3, H, W)
        rg_tensor  : torch.Tensor  shape (1, 3, H, W)
        sl_tensor  : torch.Tensor  shape (1, 3, H, W)
        image_rgb  : np.ndarray    original modality image (H, W, 3) uint8
        pred_class_idx : int       index of predicted class (0=NS1 … 3=NS4)

        Returns
        -------
        overlay_rgb : np.ndarray   heatmap blended with the modality image
        region_info : dict         structured region analysis for MedGemma
        """
        # Keep model in eval() — avoids BatchNorm batch-size-1 crash.
        # Gradients still flow via torch.enable_grad() + requires_grad_(True)
        # on the input tensor inside _compute_cam. No .train() needed.
        self.model.eval()
        cam = self._compute_cam(ant_tensor, rg_tensor, sl_tensor, pred_class_idx)

        # Resize CAM to match the input modality image
        h, w   = image_rgb.shape[:2]
        cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()

        roi_mask = self._build_roi_mask(image_rgb)
        roi_mask_f = (roi_mask > 0).astype(np.float32)
        roi_focus_score = float((cam_up * roi_mask_f).sum() / (cam_up.sum() + 1e-8))
        cam_up = cam_up * roi_mask_f
        if cam_up.max() > 0:
            cam_up = cam_up / cam_up.max()

        overlay = self._overlay_heatmap(cam_up, image_rgb, roi_mask=roi_mask)
        region_info = self._analyse_regions(cam_up)
        region_info["roi_focus_score"] = round(roi_focus_score, 3)
        region_info["roi_pixels_fraction"] = round(float(roi_mask_f.mean()), 3)
        region_info["attention_valid"] = roi_focus_score >= 0.55

        return overlay, region_info
