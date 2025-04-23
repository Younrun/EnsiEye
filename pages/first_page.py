from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from ui_components.circular_button import CircularButton
from ui_components.rounded_button import RoundedButton

events = [0, 0, 0]  # Shared state — could be imported from config

class FirstPage(Screen):
    def __init__(self, **kwargs):
        super(FirstPage, self).__init__(**kwargs)
        layout = FloatLayout()

        background = Image(source='assets/background.png', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        central_button = CircularButton(text="Start Video", size_hint=(None, None), size=(150, 150),
                                        pos_hint={'center_x': 0.5, 'center_y': 0.5})
        central_button.bind(on_press=self.start_video)

        settings_button = RoundedButton(text="Settings", size_hint=(None, None), size=(200, 50),
                                        pos_hint={'center_x': 0.5, 'y': 0.1})
        settings_button.bind(on_press=self.open_settings)

        layout.add_widget(central_button)
        layout.add_widget(settings_button)

        self.add_widget(layout)

    def start_video(self, instance):
        self.manager.current = 'video'

    def open_settings(self, instance):
        self.manager.current = 'settings'
