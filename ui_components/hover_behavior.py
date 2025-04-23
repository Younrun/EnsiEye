from kivy.properties import BooleanProperty
from kivy.core.window import Window

class HoverBehavior(object):
    hovered = BooleanProperty(False)
    border_point = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(HoverBehavior, self).__init__(**kwargs)
        self._bind_mouse_pos()

    def _bind_mouse_pos(self):
        Window.bind(mouse_pos=self.on_mouse_pos)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return
        pos = args[1]
        inside = self.collide_point(*self.to_widget(*pos))
        if self.hovered == inside:
            return
        self.hovered = inside
        if inside:
            self.on_enter()
        else:
            self.on_leave()

    def on_enter(self):
        pass

    def on_leave(self):
        pass
