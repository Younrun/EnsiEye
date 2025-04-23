from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse
from .hover_behavior import HoverBehavior

class CircularButton(Button, HoverBehavior):
    def __init__(self, **kwargs):
        super(CircularButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.color = (0, 0, 0, 1)

        with self.canvas.before:
            self.circle_color = Color(1, 1, 1, 1)
            self.ellipse = Ellipse(size=self.size, pos=self.pos)

        self.bind(pos=self.update_ellipse, size=self.update_ellipse)

    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos
        self.ellipse.size = self.size

    def on_enter(self):
        self.circle_color.rgba = (0.7, 0.7, 0.7, 1)

    def on_leave(self):
        self.circle_color.rgba = (1, 1, 1, 1)
