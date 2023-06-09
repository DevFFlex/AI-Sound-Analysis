from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty

class FileManagerScreen(Screen):
    graph_space = ObjectProperty(None)

    def __init__(self,screen_name, **kw):
        super().__init__(**kw)
        self.name = screen_name
        