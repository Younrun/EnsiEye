import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, REF_IMAGES_PATHS


class Detector:
    def __init__(self):
        # ── YOLO models ───────────────────────────────────────────────────────────
        self.model1 = YOLO("best.pt")          # vehicles & pedestrians
        self.model2 = YOLO("sign-detect.pt")   # traffic-signs

        # ── MiDaS model ──────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas  = torch.hub.load("intel-isl/MiDaS", "DPT_Large",
                                     trust_repo=True).to(self.device).eval()

        # pre-processing pipeline supplied by MiDaS
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # ───────────────────────────── public helpers ──────────────────────────────
    def detect_objects(self, frame, conf=0.3):
        """YOLO: vehicles & pedestrians."""
        return self.model1(frame, stream=True, conf=conf)

    def detect_signs(self, frame, conf=0.3):
        """YOLO: traffic signs."""
        return self.model2(frame, stream=True, conf=conf)

    def get_object_width(self, image, object_class):
        """Width (px) of a class in a reference image (if needed)."""
        for result in self.model1(image, conf=0.3):
            for box in result.boxes:
                if CLASS_NAMES[int(box.cls[0])] == object_class:
                    x1, _, x2, _ = map(int, box.xyxy[0])
                    return x2 - x1
        return None

    # ─────────────────────────── depth-estimation  ─────────────────────────────
    def estimate_depth_map(self, frame):
        """
        Returns a (H,W) numpy array in the range [0,1]
        where **0 is closest, 1 is far** (relative depth).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # dpt_transform already: divides by 255, resizes, normalises,
        # and ADDS the batch dimension -> shape (1,3,H,W)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),           # add channel dim for resize
                size=rgb.shape[:2],                # back to original H,W
                mode="bicubic",
                align_corners=False,
            ).squeeze()                            # (H,W)

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Optional smoothing – uncomment for slightly less noisy maps
        # depth = cv2.GaussianBlur(depth, (5, 5), 0)

        return depth

    def get_depth_at(self, depth_map, x, y):
        """Depth value (0–1) at pixel x,y, clamped to array bounds."""
        h, w = depth_map.shape
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        return depth_map[y, x]
