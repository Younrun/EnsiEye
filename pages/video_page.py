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
from utils.config import CLASS_NAMES, SIGN_CLASS_NAMES, KNOWN_DISTANCE, PERSON_WIDTH, CAR_WIDTH, FONT, COLORS, REF_IMAGES_PATHS

events = [0, 0, 0]  # Shared state — ideally centralized

class VideoPage(Screen):
    def __init__(self, **kwargs):
        super(VideoPage, self).__init__(**kwargs)
        self.frame_width, self.frame_height = 1920, 1080
        self.cap = None
        self.detector = Detector()

        layout = FloatLayout()
        background = KivyImage(source='assets/background.png', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)

        self.text_label = Label(text="", size_hint=(None, None), size=(300, 50),
                                pos_hint={'x': 0.03, 'y': 0.03}, font_size=20)
        layout.add_widget(self.text_label)

        back_button = CircularButton(text="<", size_hint=(None, None), size=(50, 50),
                                     pos_hint={'x': 0.05, 'top': 0.95})
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)

        self.add_widget(layout)

        # Focal lengths (set after measuring reference images)
        ref_person = cv2.imread(REF_IMAGES_PATHS['Pedestrian'])
        ref_car = cv2.imread(REF_IMAGES_PATHS['Car'])

        person_width_in_rf = self.detector.get_object_width(ref_person, 'Pedestrian')
        car_width_in_rf = self.detector.get_object_width(ref_car, 'Car')

        self.focal_person = self.detector.focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
        self.focal_car = self.detector.focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)

    def on_enter(self):
        self.cap = cv2.VideoCapture("cross.mp4")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def on_leave(self):
        Clock.unschedule(self.update_frame)
        if self.cap:
            self.cap.release()

    def go_back(self, instance):
        self.manager.current = 'first'

    def update_frame(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        # VEHICLE/PEDESTRIAN DETECTION
        results = self.detector.detect_objects(frame)
        prev_distances = {}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = CLASS_NAMES[cls]
                width = x2 - x1

                if events[1] == 1:
                    # Distance detection
                    distance = None
                    if label == 'Pedestrian':
                        distance = self.detector.distance_finder(self.focal_person, PERSON_WIDTH, width)
                    elif label == 'Car':
                        distance = self.detector.distance_finder(self.focal_car, CAR_WIDTH, width)

                    if distance:
                        if distance < 100:
                            cv2.putText(frame, 'Warning: Too Close!', (x1, y1 - 30), FONT, 0.7, COLORS['red'], 2)
                        if label in prev_distances and abs(prev_distances[label] - distance) < 1.0:
                            cv2.putText(frame, 'Slow Down', (x1, y1 - 50), FONT, 0.7, COLORS['yellow'], 2)

                        prev_distances[label] = distance

                    cv2.putText(frame, f"{label} {round(distance, 2)}cm", (x1, y1 - 10), FONT, 0.6, COLORS['green'], 1)

                elif events[1] == 0:
                    cv2.putText(frame, f"{label}", (x1 + 5, y1 + 13), FONT, 0.7, COLORS['green'], 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['green'], 2)

        # SIGN DETECTION
        if events[2] == 1:
            sign_results = self.detector.detect_signs(frame)
            for result in sign_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = SIGN_CLASS_NAMES[cls]
                    warning = f"Warning: {label.capitalize()}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['red'], 2)
                    cv2.putText(frame, warning, (x1, y1 - 10), FONT, 0.6, COLORS['red'], 2)

        # LANE DETECTION
        if events[0] == 1:
            lines = cv2.HoughLinesP(region_of_interest(canny(frame)), 2, np.pi / 180, 100, minLineLength=4, maxLineGap=50)
            if lines is not None:
                left, right = average_lines(frame, lines)
                lane_img = display_lines(frame, [left, right])
                if left is not None and right is not None:
                    fill_lane(lane_img, left, right)
                frame = cv2.addWeighted(frame, 0.9, lane_img, 1, 1)

        # Convert to Kivy texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

        self.update_label()

    def update_label(self):
        self.text_label.text = f"Lane-cruise {events[0]}  |  Distance {events[1]}  |  Signs {events[2]}"
