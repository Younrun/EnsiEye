from kivy.uix.screenmanager import Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from ui_components.rounded_button import RoundedButton
from ui_components.circular_button import CircularButton
from utils.profile_manager import save_profile, load_profile

events = [0, 0, 0]

class ProfilesPage(Screen):
    def __init__(self, **kwargs):
        super(ProfilesPage, self).__init__(**kwargs)
        layout = FloatLayout()

        background = Image(source='assets/background.png', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        self.input_text = TextInput(hint_text="Enter profile name", size_hint=(None, None), size=(300, 50),
                                    pos_hint={'center_x': 0.5, 'center_y': 0.7})

        save_btn = RoundedButton(text="Save Profile", size_hint=(None, None), size=(200, 50),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.5})
        load_btn = RoundedButton(text="Load Profile", size_hint=(None, None), size=(200, 50),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.4})
        self.message_label = Label(size_hint=(None, None), size=(400, 50),
                                   pos_hint={'center_x': 0.5, 'center_y': 0.3}, font_size=20)

        save_btn.bind(on_press=self.save_profile)
        load_btn.bind(on_press=self.load_profile)

        back_btn = CircularButton(text="<", size_hint=(None, None), size=(50, 50),
                                  pos_hint={'x': 0.05, 'top': 0.95})
        back_btn.bind(on_press=self.go_back)

        layout.add_widget(self.input_text)
        layout.add_widget(save_btn)
        layout.add_widget(load_btn)
        layout.add_widget(self.message_label)
        layout.add_widget(back_btn)

        self.add_widget(layout)

    def save_profile(self, instance):
        name = self.input_text.text.strip()
        if name:
            save_profile(name, events)
            self.message_label.text = f"✅ Profile '{name}' saved."
        else:
            self.message_label.text = "⚠️ Enter a valid name."

    def load_profile(self, instance):
        name = self.input_text.text.strip()
        if name:
            result = load_profile(name)
            if result is not None:
                global events
                events[:] = result
                self.manager.get_screen('settings').update_buttons()
                self.message_label.text = f"✅ Profile '{name}' loaded."
            else:
                self.message_label.text = f"❌ Profile '{name}' not found."
        else:
            self.message_label.text = "⚠️ Enter a profile name."

    def go_back(self, instance):
        self.manager.current = 'settings'
