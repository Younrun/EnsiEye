"""
detection/lane_detector.py
===========================

Détecteur de voies basé sur Canny + HoughLinesP, corrigé pour
attraper systématiquement les deux côtés :

• Classement des segments selon pente **ET** position relative au centre.
• Filtre des segments quasi-horizontaux.
• Remplacement “à chaud” de l’ancien code dans VideoPage.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class LaneDetector:
    # ───── paramètres ────────────────────────────────────────
    ROI_POLY    = np.array([[(200, 1080), (1100, 1080), (550, 250)]],
                           dtype=np.int32)
    CANNY_LOW, CANNY_HIGH = 50, 150
    HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH = 2, np.pi/180, 100
    MIN_LINE, MAX_GAP = 4, 50
    LINE_THICK = 10
    FILL_COLOR = (0, 255, 0)   # BGR vert

    # ────────────────────────────── public API ──────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Prend une image BGR, renvoie la même image avec overlay des deux voies.
        """
        # 1) Canny
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)

        # 2) ROI
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.ROI_POLY, 255)
        edges = cv2.bitwise_and(edges, mask)

        # 3) Hough
        lines = cv2.HoughLinesP(edges,
                                self.HOUGH_RHO,
                                self.HOUGH_THETA,
                                self.HOUGH_THRESH,
                                minLineLength=self.MIN_LINE,
                                maxLineGap=self.MAX_GAP)
        if lines is None:
            return frame

        # 4) Moyenne / classification
        left_line, right_line = self._average(frame, lines)

        # 5) Dessin
        overlay = frame.copy()
        if left_line is not None:
            x1,y1,x2,y2 = left_line
            cv2.line(overlay, (x1,y1), (x2,y2), self.FILL_COLOR, self.LINE_THICK)
        if right_line is not None:
            x1,y1,x2,y2 = right_line
            cv2.line(overlay, (x1,y1), (x2,y2), self.FILL_COLOR, self.LINE_THICK)
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], np.int32)
            cv2.fillPoly(overlay, [pts], self.FILL_COLOR)

        return cv2.addWeighted(frame, 0.9, overlay, 0.35, 0.0)

    # ───────────────────────── internal helpers ─────────────────────────
    def _average(self,
                 img: np.ndarray,
                 lines: np.ndarray
                ) -> Tuple[Optional[Tuple[int,int,int,int]],
                           Optional[Tuple[int,int,int,int]]]:
        """
        Retourne (left_line, right_line) ou (None,None).
        Chaque ligne est un tuple (x1,y1,x2,y2).
        """
        h, w = img.shape[:2]
        mid_x = w / 2.0

        left_pts, right_pts = [], []
        for ln in lines.reshape(-1,4):
            x1,y1,x2,y2 = ln
            dx, dy = x2-x1, y2-y1
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) < 0.3:   # ignore presque horizontales
                continue
            intercept = y1 - slope*x1
            line_mid = (x1 + x2) / 2.0

            # classification sur signe de pente ET position
            if slope < 0 and line_mid < mid_x:
                left_pts.append((slope, intercept))
            elif slope > 0 and line_mid > mid_x:
                right_pts.append((slope, intercept))

        # si l’un manque, on renvoie None
        if not left_pts or not right_pts:
            return None, None

        # moyennes des paramètres (slope, intercept)
        left_avg  = np.mean(left_pts,  axis=0)
        right_avg = np.mean(right_pts, axis=0)

        def make_coords(params):
            k, b = params
            y1, y2 = h, int(h*0.6)
            x1 = int((y1 - b)/k)
            x2 = int((y2 - b)/k)
            return (x1, y1, x2, y2)

        return make_coords(left_avg), make_coords(right_avg)
