from kivy.app import App
from kivy.uix.screenmanager import ScreenManager

from pages.first_page import FirstPage
from pages.video_page import VideoPage
from pages.settings_page import SettingsPage
from pages.profiles_page import ProfilesPage

class MainApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(FirstPage(name='first'))
        sm.add_widget(VideoPage(name='video'))
        sm.add_widget(SettingsPage(name='settings'))
        sm.add_widget(ProfilesPage(name='profiles'))
        return sm

if __name__ == '__main__':
    MainApp().run()
