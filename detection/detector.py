"""
detection/detector.py
─────────────────────
• YOLOv8 (best.pt)   : vehicles / pedestrians
• YOLOv8 (sign*.pt)  : traffic signs
• MiDaS  DPT_Hybrid  : depth map in [0,1]

Robust to PyTorch 2.6 ‘weights_only’ behaviour, avoids GPU OOM during
Conv+BN fusion, avoids FP16 bicubic‑upsample crash.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES

# ─── Torch‑2.6 “weights_only” patch (old behaviour) ─────────────────────────
torch.load = (lambda old=torch.load:
              lambda *a, **k: old(*a, **{**k, "weights_only": False}))()

class Detector:
    def __init__(self):
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Detector] device: {self.device}")

        # ── 1. Load YOLO on CPU, fuse, then move ────────────────────────────
        self.model1 = YOLO("best.pt")            # vehicles / people
        self.model2 = YOLO("sign-detect.pt")     # traffic signs

        # fuse Conv+BN once (CPU → uses system RAM)
        self.model1.model.fuse()
        self.model2.model.fuse()

        # move fused nets to GPU/CPU (keep FP32 for stability & RAM)
        self.model1.model.to(self.device)
        self.model2.model.to(self.device)

        # ── 2. MiDaS depth (fast) ───────────────────────────────────────────
        self.midas = (torch.hub
                      .load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
                      .to(self.device).eval())
        if self.device.type == "cuda":
            self.midas.half()  # FP16 for speed

        self.transform = torch.hub.load("intel-isl/MiDaS",
                                        "transforms").dpt_transform

    # ─────────── YOLO wrappers (already fused, no extra args) ───────────────
    def detect_objects(self, frame, conf: float = 0.3):
        """Vehicles / pedestrians."""
        return self.model1(frame, stream=True, conf=conf)

    def detect_signs(self, frame, conf: float = 0.3):
        """Traffic signs."""
        return self.model2(frame, stream=True, conf=conf)

    # ─────────── Depth helpers ──────────────────────────────────────────────
    def estimate_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """Return (H,W) depth in [0,1]. 0 ≈ near, 1 ≈ far."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)
        if self.device.type == "cuda":
            batch = batch.half()

        with torch.no_grad():
            pred = self.midas(batch)             # (1, H', W')
            if pred.dtype == torch.float16:      # safer upsample in FP32
                pred = pred.float()
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False).squeeze()

        depth = pred.cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def get_depth_at(self, depth_map: np.ndarray, x: int, y: int) -> float:
        h, w = depth_map.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(depth_map[y, x])

    # (optional) width helper
    def get_object_width(self, image: np.ndarray, obj_class: str):
        for res in self.model1(image, conf=0.3):
            for box in res.boxes:
                if CLASS_NAMES[int(box.cls[0])] == obj_class:
                    x1, _, x2, _ = map(int, box.xyxy[0])
                    return x2 - x1
        return None
