import cv2
import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from ui_components.circular_button import CircularButton
from detection.detector import Detector
from detection.lane_detector import (
    canny, region_of_interest, average_lines, display_lines, fill_lane
)
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, FONT, COLORS

events = [0, 1, 1]           # Lane, Distance, Sign
DEPTH_NEAR   = 0.10          # < 0.10  ⇒ red   “too close”
DEPTH_MID    = 0.30          # 0.10-0.30 ⇒ yellow “medium”
# > 0.30 ⇒ green “safe”

class VideoPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_width, self.frame_height = 1920, 1080
        self.cap      = None
        self.detector = Detector()

        layout      = FloatLayout()
        background  = KivyImage(source='assets/background.png',
                                allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)

        self.text_label = Label(text="", size_hint=(None, None), size=(300, 50),
                                pos_hint={'x': 0.03, 'y': 0.03}, font_size=20)
        layout.add_widget(self.text_label)

        back_btn = CircularButton(text="<", size_hint=(None, None), size=(50, 50),
                                  pos_hint={'x': 0.05, 'top': 0.95})
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)

        self.add_widget(layout)

    # ───────────────────────── screen life-cycle ──────────────────────────
    def on_enter(self):
        self.cap = cv2.VideoCapture("cross.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        Clock.schedule_interval(self.update_frame, 1 / 30)

    def on_leave(self):
        Clock.unschedule(self.update_frame)
        if self.cap:
            self.cap.release()

    def go_back(self, *_):
        self.manager.current = "first"

    # ───────────────────────── helpers ────────────────────────────────────
    def is_inside_roi(self, x, y):
        h = self.frame_height
        roi = np.array([[(200, h), (1100, h), (550, 250)]], np.int32)
        return cv2.pointPolygonTest(roi[0], (x, y), False) >= 0

    # ───────────────────────── main loop ──────────────────────────────────
    def update_frame(self, _dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        depth_map = self.detector.estimate_depth_map(frame)        # (H,W) 0-1
        results   = self.detector.detect_objects(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy         = (x1 + x2) // 2, (y1 + y2) // 2
                if not self.is_inside_roi(cx, cy):
                    continue

                label = CLASS_NAMES[int(box.cls[0])]
                depth = self.detector.get_depth_at(depth_map, cx, cy)

                # 1️⃣  print to terminal
                print(f"[Depth] {label:<10} at ({cx:4},{cy:4})  -> {depth:.3f}")

                # 2️⃣  choose colour
                if depth < DEPTH_NEAR:
                    colour = (0, 0, 255)        # red   BGR
                    cv2.putText(frame, "Too close!", (x1, y1 - 25),
                                FONT, 0.7, colour, 2)
                elif depth < DEPTH_MID:
                    colour = (0, 255, 255)      # yellow
                else:
                    colour = (0, 255, 0)        # green

                # 3️⃣  draw box & label with depth value
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{label} {depth:.2f}",
                            (x1, y1 - 5), FONT, 0.6, colour, 2)

        # ───────── optional: traffic-signs & lane code (unchanged) ─────────
        if events[2]:
            for result in self.detector.detect_signs(frame):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    depth = self.detector.get_depth_at(depth_map , cx , cy)
                    print(f"[Depth] Sign {SIGN_CLASS_NAMES[int(box.cls[0])]} "
                  f"depth={depth:.3f}")
                    if not self.is_inside_roi(cx, cy):
                        continue
                    sign   = SIGN_CLASS_NAMES[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["red"], 2)
                    cv2.putText(frame, f"Sign: {sign}", (x1, y1 - 10),
                                FONT, 0.6, COLORS["red"], 2)

        if events[0]:
            lines = cv2.HoughLinesP(region_of_interest(canny(frame)),
                                    2, np.pi / 180, 100,
                                    minLineLength=4, maxLineGap=50)
            if lines is not None:
                left, right = average_lines(frame, lines)
                lane_img    = display_lines(frame, [left, right])
                if left is not None and right is not None:
                    fill_lane(lane_img, left, right)
                frame = cv2.addWeighted(frame, 0.9, lane_img, 1, 1)

        # ───────── send frame to Kivy texture ─────────
        buf      = cv2.flip(frame, 0).tobytes()
        texture  = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image_widget.texture = texture

        self.update_label()

    def update_label(self):
        self.text_label.text = f"Lane {events[0]} | Dist {events[1]} | Sign {events[2]}"
