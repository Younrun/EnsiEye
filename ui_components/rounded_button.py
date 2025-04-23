from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle
from .hover_behavior import HoverBehavior

class RoundedButton(Button, HoverBehavior):
    def __init__(self, **kwargs):
        super(RoundedButton, self).__init__(**kwargs)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        self.color = (0, 0, 0, 1)
        self.active = False

        with self.canvas.before:
            self.rect_color = Color(1, 1, 1, 1)
            self.rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[20])

        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_enter(self):
        if not self.active:
            self.rect_color.rgba = (0.7, 0.7, 0.7, 1)

    def on_leave(self):
        if not self.active:
            self.rect_color.rgba = (1, 1, 1, 1)

    def set_active(self, active):
        self.active = active
        if active:
            self.rect_color.rgba = (0, 1, 0, 1)
        else:
            self.rect_color.rgba = (1, 1, 1, 1)
