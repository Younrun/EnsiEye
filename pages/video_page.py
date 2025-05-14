import cv2
import numpy as np
from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from ui_components.circular_button import CircularButton
from detection.detector      import Detector
from detection.lane_detector import LaneDetector
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, FONT, COLORS

events = [1, 1, 1]              # [lane, distance, sign]
DEPTH_NEAR, DEPTH_MID = 0.10, 0.30


class VideoPage(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)

        self.frame_width, self.frame_height = 1920, 1080
        self.cap = None
        self.detector   = Detector()
        self.lane_det   = LaneDetector()

        # ---------- UI ----------------------------------------------------
        lay = FloatLayout()
        lay.add_widget(KivyImage(source="assets/background.png",
                                 allow_stretch=True, keep_ratio=False))

        self.image_widget = KivyImage()
        lay.add_widget(self.image_widget)

        self.text_label = Label(size_hint=(None, None), size=(350, 50),
                                pos_hint={"x": .03, "y": .03}, font_size=20)
        lay.add_widget(self.text_label)

        back = CircularButton(text="<", size_hint=(None, None), size=(50, 50),
                              pos_hint={"x": .05, "top": .95})
        back.bind(on_press=lambda *_: setattr(self.manager, "current", "first"))
        lay.add_widget(back)
        self.add_widget(lay)

    # ---------- life‑cycle ----------------------------------------------
    def on_enter(self):
        self.cap = cv2.VideoCapture("istockphoto-1313165564-640_adpp_is.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        Clock.schedule_interval(self.update_frame, 1/30)

    def on_leave(self):
        Clock.unschedule(self.update_frame)
        if self.cap:
            self.cap.release()

    # ---------- main loop ------------------------------------------------
    def update_frame(self, _dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 1. lane overlay
        if events[0]:
            frame = self.lane_det.process(frame)

        # 2. depth once
        depth_map = self.detector.estimate_depth_map(frame)

        # 3. vehicles / pedestrians
        if events[1]:
            self._draw_objects(frame, depth_map)

        # 4. traffic signs
        if events[2]:
            self._draw_signs(frame, depth_map)

        # 5. send to Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image_widget.texture = tex

        self.text_label.text = f"Lane {events[0]} | Dist {events[1]} | Sign {events[2]}"

    # ---------- drawing helpers -----------------------------------------
    def _draw_objects(self, frame, depth_map):
        for res in self.detector.detect_objects(frame):
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2   # centre du box

                lbl = CLASS_NAMES[int(box.cls[0])]
                d   = self.detector.get_depth_at(depth_map, cx, cy)
                print(f"[Depth] {lbl:<10} ({cx},{cy}) -> {d:.3f}")

                # couleur selon la profondeur
                if d < DEPTH_NEAR:
                    colour, warn = (0, 0, 255), "Too close!"
                elif d < DEPTH_MID:
                    colour, warn = (0, 255, 255), None
                else:
                    colour, warn = (0, 255, 0), None

                if warn:
                    cv2.putText(frame, warn, (x1, y1 - 25), FONT, .7, colour, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{lbl} {d:.2f}", (x1, y1 - 5), FONT, .6, colour, 2)

    def _draw_signs(self, frame, depth_map):
        for res in self.detector.detect_signs(frame):
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                sign = SIGN_CLASS_NAMES[int(box.cls[0])]
                d    = self.detector.get_depth_at(depth_map, cx, cy)
                print(f"[Depth] Sign {sign:<8} -> {d:.3f}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["red"], 2)
                cv2.putText(frame, sign, (x1, y1 - 10), FONT, .6, COLORS["red"], 2)
