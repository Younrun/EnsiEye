"""
LaneDetector (fast, classical)
──────────────────────────────
Detects lane area and overlays a filled green polygon.

Key optimisations
• Work at 640×360 internal resolution
• Full sliding‑window only on first frame or when tracking lost
• Fast margin search each subsequent frame
• Exponential moving‑average smoothing
"""

import cv2, numpy as np, time
from collections import deque

#  ───────── tunables ──────────────────────────────────────────────────────
PROC_W, PROC_H = 640, 360          # internal working resolution (16:9)
WINDOWS, MARGIN, MINPIX = 9, 60, 50
SMOOTH_ALPHA = 0.7                 # 0=no smooth, 1=freeze
FPS_LOG_EVERY = 200

#  Perspective matrices (scale from 1080p coordinates)
_src = np.float32([[200,720],[1100,720],[595,450],[685,450]]) * PROC_W/1280
_dst = np.float32([[150,360],[490,360],[150,0],[490,0]])       * PROC_W/640
M    = cv2.getPerspectiveTransform(_src, _dst)
MINV = cv2.getPerspectiveTransform(_dst, _src)

#  ───────── inexpensive binary mask ───────────────────────────────────────
def binary_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 30, 110), (40, 255, 255))
    white  = cv2.inRange(hsv, (0, 0, 200),   (255, 40, 255))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_eq = cv2.createCLAHE(2.0, (8,8)).apply(lab[:,:,0])
    sobel = cv2.Sobel(l_eq, cv2.CV_16S, 1, 0, ksize=3)
    grad  = cv2.inRange(np.abs(sobel).astype(np.uint8), 30, 150)

    mask = cv2.bitwise_or(yellow, white)
    mask = cv2.bitwise_or(mask, grad)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),2)
    return mask // 255

#  ───────── LaneDetector class ────────────────────────────────────────────
class LaneDetector:
    def __init__(self):
        self.left_fit = self.right_fit = None
        self.ema_left = self.ema_right = None
        self.frame_ct, self.t_last = 0, time.time()

    #  public --------------------------------------------------------------
    def process(self, frame):
        self.frame_ct += 1
        small  = cv2.resize(frame, (PROC_W, PROC_H))
        warped = cv2.warpPerspective(binary_mask(small), M, (PROC_W, PROC_H))

        if self.left_fit is None or self.right_fit is None:
            self.left_fit, self.right_fit = self._sliding_window(warped)
        else:
            ok = self._margin_search(warped)
            if not ok:
                self.left_fit, self.right_fit = self._sliding_window(warped)

        if self.left_fit is None or self.right_fit is None:
            return frame

        # EMA smoothing
        if self.ema_left is None:
            self.ema_left, self.ema_right = self.left_fit, self.right_fit
        else:
            self.ema_left  = SMOOTH_ALPHA*self.ema_left  + (1-SMOOTH_ALPHA)*self.left_fit
            self.ema_right = SMOOTH_ALPHA*self.ema_right + (1-SMOOTH_ALPHA)*self.right_fit

        lane_poly = self._poly_points(PROC_H, self.ema_left, self.ema_right)
        #───── Guard against NaN / empty poly──────────────────────────────
        if (lane_poly is None or lane_poly.size<6 or np.isnan(lane_poly).all()):
            return frame #we skip the frame but keeping the detector state
        pts = lane_poly.astype(np.int32)
            # split into left / right arrays for nice walls
        h = small.shape[0]
        ploty = np.linspace(0, h - 1, h)

        left  = ((self.ema_left[0] * ploty + self.ema_left[1]) * ploty +
                self.ema_left[2]).astype(np.int32)
        
        right = ((self.ema_right[0] * ploty + self.ema_right[1]) * ploty +
                self.ema_right[2]).astype(np.int32)
        
        pts_left  = np.column_stack((left , ploty.astype(np.int32)))
        pts_right = np.column_stack((right, ploty.astype(np.int32)))[::-1]

        overlay   = np.zeros_like(small, dtype=np.uint8)


        for i in range(len(ploty) - 1):
            a = 0.4 * (1 - i / len(ploty))          # near = 0.4, far → 0
            quad = np.array([pts_left[i],
                            pts_left[i + 1],
                            pts_right[i + 1],
                            pts_right[i]], np.int32)
            cv2.fillPoly(overlay, [quad], color=(0, int(255 * a), 0))


        cv2.polylines(overlay, [pts_left],  False, (0, 255, 255), 4)
        cv2.polylines(overlay, [pts_right], False, (0, 255, 255), 4)


        unwarp    = cv2.warpPerspective(overlay, MINV, (PROC_W, PROC_H))
        blended   = cv2.addWeighted(small, 1, unwarp, 1, 0)
        result    = cv2.resize(blended, (frame.shape[1], frame.shape[0]))

        if self.frame_ct % FPS_LOG_EVERY == 0:
            now = time.time()
            print(f"[Lane] {FPS_LOG_EVERY/(now-self.t_last):.1f} FPS (fast)")
            self.t_last = now
        return result

    #  helpers -------------------------------------------------------------
    def _sliding_window(self, binary):
        h, w = binary.shape
        hist = np.sum(binary[h//2:], axis=0)
        mid  = w//2
        lx_base = np.argmax(hist[:mid]); rx_base = np.argmax(hist[mid:]) + mid

        nz_y, nz_x = binary.nonzero()
        win_h = h // WINDOWS
        lx_cur, rx_cur = lx_base, rx_base
        left_inds, right_inds = [], []

        for win in range(WINDOWS):
            y_low, y_high = h-(win+1)*win_h, h-win*win_h
            lx_low, lx_high = lx_cur-MARGIN, lx_cur+MARGIN
            rx_low, rx_high = rx_cur-MARGIN, rx_cur+MARGIN
            good_left  = ((nz_y>=y_low)&(nz_y<y_high)&
                          (nz_x>=lx_low)&(nz_x<lx_high)).nonzero()[0]
            good_right = ((nz_y>=y_low)&(nz_y<y_high)&
                          (nz_x>=rx_low)&(nz_x<rx_high)).nonzero()[0]
            left_inds.append(good_left);  right_inds.append(good_right)
            if good_left.size  > MINPIX: lx_cur = int(np.mean(nz_x[good_left]))
            if good_right.size > MINPIX: rx_cur = int(np.mean(nz_x[good_right]))

        left_inds  = np.concatenate(left_inds)
        right_inds = np.concatenate(right_inds)
        if left_inds.size < 400 or right_inds.size < 400:
            return None, None
        left_fit  = np.polyfit(nz_y[left_inds],  nz_x[left_inds],  2)
        right_fit = np.polyfit(nz_y[right_inds], nz_x[right_inds], 2)
        return left_fit, right_fit

    def _margin_search(self, binary):
        h = binary.shape[0]
        nz_y, nz_x = binary.nonzero()
        left_zone  = (nz_x > (self.left_fit[0]*nz_y**2 + self.left_fit[1]*nz_y +
                              self.left_fit[2] - MARGIN)) & \
                     (nz_x < (self.left_fit[0]*nz_y**2 + self.left_fit[1]*nz_y +
                              self.left_fit[2] + MARGIN))
        right_zone = (nz_x > (self.right_fit[0]*nz_y**2 + self.right_fit[1]*nz_y +
                              self.right_fit[2] - MARGIN)) & \
                     (nz_x < (self.right_fit[0]*nz_y**2 + self.right_fit[1]*nz_y +
                              self.right_fit[2] + MARGIN))
        lx, ly = nz_x[left_zone],  nz_y[left_zone]
        rx, ry = nz_x[right_zone], nz_y[right_zone]
        if lx.size < 200 or rx.size < 200:
            return False
        self.left_fit  = np.polyfit(ly, lx, 2)
        self.right_fit = np.polyfit(ry, rx, 2)
        return True

    @staticmethod
    def _poly_points(h, lf, rf):
        ploty = np.linspace(0, h-1, h)
        left  = lf[0]*ploty**2 + lf[1]*ploty + lf[2]
        right = rf[0]*ploty**2 + rf[1]*ploty + rf[2]
        left_pts  = np.transpose(np.vstack([left,  ploty]))
        right_pts = np.flipud(np.transpose(np.vstack([right, ploty])))
        return np.hstack([left_pts, right_pts]).astype(np.int32)
