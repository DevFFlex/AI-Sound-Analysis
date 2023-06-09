import io
import numpy
import threading
import time
import random
import datetime
import easygui
import os
from tensorflow import keras
import easygui

import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.garden.graph import Graph, MeshLinePlot,LinePlot
import math
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from kivy.base import runTouchApp
from kivy.graphics import Color, Line
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.modalview import ModalView

from file.main_screen import MainScreen
from file.file_manager import FileManagerScreen


class ScreenManagerApp(App):

    def build(self):

        Window.size = (1200, 600)
        Window.minimum_width = 800
        Window.minimum_height = 600
        Window.fullscreen = 'auto'

        Builder.load_file('file/main_screen.kv')
        Builder.load_file('file/file_manager.kv')

        sm = ScreenManager()
        sm.add_widget(MainScreen(screen_name='MainScreen'))
        sm.add_widget(FileManagerScreen(screen_name='FileManagerScreen'))
        return sm
    
    def changeto_filemanagerScreen(self):
        self.root.current = 'FileManagerScreen'
    
    def on_stop(self):
        print("แอปพลิเคชันกำลังปิดลง")
        main_screen : MainScreen = self.root.get_screen('MainScreen')
        main_screen.variable.app_running = False


sm = ScreenManagerApp()
sm.run()




