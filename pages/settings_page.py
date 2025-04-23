from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from ui_components.rounded_button import RoundedButton
from ui_components.circular_button import CircularButton

events = [0, 0, 0]

class SettingsPage(Screen):
    def __init__(self, **kwargs):
        super(SettingsPage, self).__init__(**kwargs)
        layout = FloatLayout()

        background = Image(source='assets/background.png', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        self.lane_cruise_button = RoundedButton(text="Lane Cruise", size_hint=(None, None), size=(200, 50),
                                                pos_hint={'center_x': 0.5, 'center_y': 0.7})
        self.distance_detection_button = RoundedButton(text="Distance Detection", size_hint=(None, None), size=(200, 50),
                                                       pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.traffic_sign_button = RoundedButton(text="Traffic Sign Detection", size_hint=(None, None), size=(200, 50),
                                                 pos_hint={'center_x': 0.5, 'center_y': 0.3})
        profile_button = RoundedButton(text="Profiles", size_hint=(None, None), size=(200, 50),
                                       pos_hint={'center_x': 0.5, 'center_y': 0.1})

        self.lane_cruise_button.bind(on_press=self.toggle_lane)
        self.distance_detection_button.bind(on_press=self.toggle_distance)
        self.traffic_sign_button.bind(on_press=self.toggle_sign)
        profile_button.bind(on_press=self.open_profiles)

        back_button = CircularButton(text="<", size_hint=(None, None), size=(50, 50),
                                     pos_hint={'x': 0.05, 'top': 0.95})
        back_button.bind(on_press=self.go_back)

        layout.add_widget(self.lane_cruise_button)
        layout.add_widget(self.distance_detection_button)
        layout.add_widget(self.traffic_sign_button)
        layout.add_widget(profile_button)
        layout.add_widget(back_button)

        self.add_widget(layout)
        self.update_buttons()

    def toggle_lane(self, instance):
        events[0] = 1 - events[0]
        self.update_buttons()

    def toggle_distance(self, instance):
        events[1] = 1 - events[1]
        self.update_buttons()

    def toggle_sign(self, instance):
        events[2] = 1 - events[2]
        self.update_buttons()

    def update_buttons(self):
        self.lane_cruise_button.set_active(events[0] == 1)
        self.distance_detection_button.set_active(events[1] == 1)
        self.traffic_sign_button.set_active(events[2] == 1)

    def go_back(self, instance):
        self.manager.current = 'first'

    def open_profiles(self, instance):
        self.manager.current = 'profiles'
