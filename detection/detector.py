import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, REF_IMAGES_PATHS

class Detector:
    def __init__(self):
        # ── pick device ─────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str  = "cuda" if self.device.type == "cuda" else "cpu"
        print(f"[Detector] running on device: {self.device}")

        # ── YOLO models on GPU/CPU ───────────────────────────────────────────────
        # pass `device=` so Ultralytics uses the right backend
        self.model1 = YOLO("best.pt",        device=device_str)  # vehicles & pedestrians
        self.model2 = YOLO("sign-detect.pt", device=device_str)  # traffic-signs

        # ── optional: half-precision on CUDA for extra speed ──────────────────
        if self.device.type == "cuda":
            self.model1.model.half()
            self.model2.model.half()

        # ── MiDaS depth model on GPU/CPU ────────────────────────────────────────
        # Switched to DPT_Large—change to "DPT_Hybrid" or "MiDaS_small" if you want lighter
        self.midas = (
            torch.hub
                 .load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
                 .to(self.device)
                 .eval()
        )
        if self.device.type == "cuda":
            self.midas = self.midas.half()

        # ── MiDaS pre-processing pipeline ───────────────────────────────────────
        # This transform normalizes, resizes, and adds batch dim (1,3,H,W)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # ───────────────────────────── public helpers ──────────────────────────────
    def detect_objects(self, frame, conf=0.3):
        """YOLO inference for vehicles & pedestrians."""
        return self.model1(frame, stream=True, conf=conf)

    def detect_signs(self, frame, conf=0.3):
        """YOLO inference for traffic signs."""
        return self.model2(frame, stream=True, conf=conf)

    def get_object_width(self, image, object_class):
        """Helper: pixel width of an object in a reference image."""
        for result in self.model1(image, conf=0.3):
            for box in result.boxes:
                if CLASS_NAMES[int(box.cls[0])] == object_class:
                    x1, _, x2, _ = map(int, box.xyxy[0])
                    return x2 - x1
        return None

    # ───────────────────────── depth-estimation ────────────────────────────────
    def estimate_depth_map(self, frame):
        """
        Returns a (H,W) numpy array in [0,1] where 0 is closest and 1 is farthest.
        """
        # Convert BGR→RGB and prepare batch
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)
        if self.device.type == "cuda":
            batch = batch.half()  # FP16 on GPU

        with torch.no_grad():
            pred = self.midas(batch)  # (1, H', W')
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),          # → (1,1,H',W')
                size=rgb.shape[:2],         # back to (H,W)
                mode="bicubic",
                align_corners=False
            ).squeeze()                    # → (H,W)

        depth = pred.cpu().numpy()
        # normalize to [0,1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def get_depth_at(self, depth_map, x, y):
        """Depth value (0–1) at pixel (x,y), clamped to bounds."""
        h, w = depth_map.shape
        x = min(max(x, 0), w-1)
        y = min(max(y, 0), h-1)
        return depth_map[y, x]
