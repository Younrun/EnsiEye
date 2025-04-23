import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, REF_IMAGES_PATHS

class Detector:
    def __init__(self):
        # Load YOLO models
        self.model1 = YOLO('best.pt')  # Vehicles & pedestrians
        self.model2 = YOLO('sign-detect.pt')  # Traffic signs

        # Load MiDaS model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()

        # MiDaS transform
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def detect_objects(self, frame, conf=0.3):
        return self.model1(frame, stream=True, conf=conf)

    def detect_signs(self, frame, conf=0.3):
        return self.model2(frame, stream=True, conf=conf)

    def get_object_width(self, image, object_class):
        results = self.model1(image, conf=0.3)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if CLASS_NAMES[cls] == object_class:
                    return int(box.xyxy[0][2] - box.xyxy[0][0])
        return None

    def estimate_depth_map(self, frame):
        """Estimate depth map using MiDaS"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_tensor.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def get_depth_at(self, depth_map, x, y):
        """Get depth value at pixel (x, y)"""
        h, w = depth_map.shape
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)
        return depth_map[y, x]
