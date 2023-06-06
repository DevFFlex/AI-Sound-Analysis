# from _data_manager import DataManager
from _sound_io import SoundIO
from _variable import Variable
from _ai_class import AI

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


class MyDialog(ModalView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (400, 200)

        # สร้างองค์ประกอบภายใน Dialog
        self.content = Button(text='Hello, I am a dialog!', size_hint=(0.8, 0.5))

        # ปุ่มเพื่อปิด Dialog
        self.close_button = Button(text='Close', size_hint=(0.4, 0.3), pos_hint={'center_x': 0.5})
        self.close_button.bind(on_release=self.dismiss)

        # เพิ่มองค์ประกอบใน Dialog
        self.add_widget(self.content)
        self.add_widget(self.close_button)


class FileManagerScreen(Screen):
    graph_space = ObjectProperty(None)

    def __init__(self,screen_name, **kw):
        super().__init__(**kw)
        self.name = screen_name
        
    
    

class MainScreen(Screen):
    dataset_inputclassname = ObjectProperty(None)
    dataset_textoutput = ObjectProperty(None)
    dataset_btnstart = ObjectProperty(None)
    dataset_inputdatacount = ObjectProperty(None)
    dataset_switchsavedata = ObjectProperty(None)
    dataset_switchautotrain = ObjectProperty(None)

    sa_btnrecord = ObjectProperty(None)
    sa_btnplaylastsound = ObjectProperty(None)
    sa_gensound_freq = ObjectProperty(None)
    sa_gensound_sec = ObjectProperty(None)
    sa_gensound_btnplay = ObjectProperty(None)
    sa_filter_inputamp = ObjectProperty(None)
    sa_btnsavedata = ObjectProperty(None)
    sa_btnloaddata = ObjectProperty(None)
    sa_filter_inputspace = ObjectProperty(None)

    train_btntrain = ObjectProperty(None)
    train_inputclasscount = ObjectProperty(None)
    train_textoutput = ObjectProperty(None)
    train_inputepoch = ObjectProperty(None)

    predict_btnpredict = ObjectProperty(None)
    predict_class = ObjectProperty(None)
    predict_maxconfi = ObjectProperty(None)
    predict_swautopredict = ObjectProperty(None)

    model_model = ObjectProperty(None)
    model_testaccuracy = ObjectProperty(None)
    model_testloss = ObjectProperty(None)

    file_btnscandataset = ObjectProperty(None)
    file_filecount = ObjectProperty(None)
    file_traincount = ObjectProperty(None)
    file_validcount = ObjectProperty(None)
    file_testcount = ObjectProperty(None)
    file_classcount = ObjectProperty(None)
    file_inputtrain_persen = ObjectProperty(None)
    file_btnopenfolder = ObjectProperty(None)

    graph_space = ObjectProperty(None)
    graph_spaceOne = ObjectProperty(None)
    graph_inputMinScaleX = ObjectProperty(None)
    graph_inputMaxScaleX = ObjectProperty(None)
    graph_inputMinScaleY = ObjectProperty(None)
    graph_inputMaxScaleY = ObjectProperty(None)

    output = ObjectProperty(None)


    def trainCallbackFunction(self, epoch, logs):
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        self.train_textoutput.text = f"epoch {epoch} | Accuracy = {'%.2f' % accuracy} | loss = {'%.2f' % loss}"

    

    def __init__(self,screen_name, **kwargs):
        super().__init__(**kwargs)
        self.name = screen_name

        self.variable = Variable()
        self.sound_io = SoundIO()
        self.ai = AI(self)

        self.dataset_btnstart.bind(on_press=lambda btn: threading.Thread(target=self.dataset_onbtnstart,args=(btn,)).start())  
        self.train_btntrain.bind(on_press = lambda btn : threading.Thread(target=self.train_onbtnclick,args=(btn,)).start())
        self.predict_btnpredict.bind(on_press = lambda btn : threading.Thread(target=self.predict_onbtnclick,args=(btn,)).start())
        self.file_btnscandataset.bind(on_press = lambda btn : threading.Thread(target=self.filezone_onClickProcessDataset,args=(btn,)).start())
        self.file_btnopenfolder.bind(on_press = lambda btn : easygui.fileopenbox())

        self.sa_btnrecord.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickRecord,args=(btn,)).start())
        self.sa_btnplaylastsound.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickPlayLastRecord,args=(btn,)).start())
        self.sa_gensound_btnplay.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickPlayGenSound,args=(btn,)).start())
        
    


        def onDialog(btn):
            dialog = MyDialog()
            dialog.open()

        # self.sa_btnrecord.bind(on_press=onDialog)


        def update_label(dt):
            
            self.model_model.color = [0,1,0,1] if self.variable.MODEL != None else [1,0,0,1]
            self.model_model.text = "Model Is Ready" if self.variable.MODEL != None else "No Model"
            
            if self.variable.TestAccuracy != None and self.variable.TestLoss != None:
                self.model_testaccuracy.text = str("%.2f" % (self.variable.TestAccuracy * 100) + "%")
                self.model_testloss.text = str("%.2f" % (self.variable.TestLoss * 100) + "%")

            self.file_filecount.text = str(self.ai.getDataFileCount())

            self.ai.createBaseResource()

            if self.variable.DataSet.isData():
                self.train_btntrain.disabled = False
            else:
                self.train_btntrain.disabled = True


            if self.ai.isModel():
                self.predict_btnpredict.disabled = False
            else:
                self.predict_btnpredict.disabled = True


        Clock.schedule_interval(lambda dt: update_label(dt,), 1)

        

        # self.btnSelectMIC.bind(on_release=self.dropdown.open)
        # self.dropdown = DropDown()
        # self.add_dropdown_options()
        # self.dropdown.bind(on_select=lambda instance, x: setattr(self.btnSelectMIC, 'text', x))
        # self.dropdown.bind(on_open=self.update_dropdown_options)

    """def add_dropdown_options(self):
        for index in range(10):
            btn = Button(text='Value %d' % random.randint(0,1000), size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

    def update_dropdown_options(self, instance):  
        self.dropdown.clear_widgets()
        self.add_dropdown_options()

    def show_dropdown(self, instance):
        self.dropdown.open(instance)"""

    def stop_autopredict(self,btn):
        self.variable.status_loop_autopredict = False    

    def plotGraph(self,freqamp,filterAMP):
        freq,amp = freqamp
        data = list(zip(freq, amp))
        
        DATA_FORMATT = "((1,2),(5,6),...)"

        XLABEL = 'Frequency (Hz)'
        YLABEL = 'Amplitude (db)'
        GRAPH_X_RANGE = [0,3000]
        GRAPH_Y_RANGE = [0,100]

        xminrange_text = self.graph_inputMinScaleX.text
        xmaxrange_text = self.graph_inputMaxScaleX.text
        yminrange_text = self.graph_inputMinScaleY.text
        ymaxrange_text = self.graph_inputMaxScaleY.text

        if xminrange_text != "" and xminrange_text.isdigit():
            GRAPH_X_RANGE[0] = int(xminrange_text)
        
        if xmaxrange_text != "" and xmaxrange_text.isdigit():
            GRAPH_X_RANGE[1] = int(xmaxrange_text)
        
        if yminrange_text != "" and yminrange_text.isdigit():
            GRAPH_Y_RANGE[0] = int(yminrange_text)
        
        if ymaxrange_text != "" and ymaxrange_text.isdigit():
            GRAPH_Y_RANGE[1] = int(ymaxrange_text)

        self.graph_inputMinScaleX.text = str(GRAPH_X_RANGE[0])
        self.graph_inputMaxScaleX.text = str(GRAPH_X_RANGE[1])
        self.graph_inputMinScaleY.text = str(GRAPH_Y_RANGE[0])
        self.graph_inputMaxScaleY.text = str(GRAPH_Y_RANGE[1])

        
        
        
        self.graph_space.clear_widgets()
        
        self.variable.graphHistory.addPoint(data)

        self.graph = Graph(xlabel=XLABEL, ylabel=YLABEL, x_ticks_minor=5,
                      x_ticks_major=(GRAPH_X_RANGE[1] - GRAPH_X_RANGE[0]) / 15, y_ticks_major=(GRAPH_Y_RANGE[1] - GRAPH_Y_RANGE[0]) / 10,
                      y_grid_label=True, x_grid_label=True, padding=0,
                      x_grid=True, y_grid=True, xmin=GRAPH_X_RANGE[0], xmax=GRAPH_X_RANGE[1], ymin=GRAPH_Y_RANGE[0], ymax=GRAPH_Y_RANGE[1])
        
        i = 0
        for points_item in self.variable.graphHistory.getGraphPoint():
            new_plot = LinePlot(color= self.variable.graphHistory.GRAPH_COLOR[i])
            new_plot.points = points_item
            self.graph.add_plot(new_plot)
            i+=1

        if filterAMP != "" and filterAMP.isdigit():
            new_plot = LinePlot(color= [0, 0, 1, 1])
            new_plot.points = [(0,int(filterAMP)),(3000,int(filterAMP))]
            self.graph.add_plot(new_plot)
        
        self.graph_space.add_widget(self.graph)

        self.output.text = f'vdfc = {self.variable.datafilecount}   ai = {self.ai.getDataFileCount()}'
        if self.variable.datafilecount != self.ai.getDataFileCount():
            self.variable.datafilecount = self.ai.getDataFileCount()
            class0,class1 = self.ai.getAverageClass()

            data_graphPlot_class0 = list(zip(freq, class0))
            data_graphPlot_class1 = list(zip(freq, class1))

            self.graph_spaceOne.clear_widgets()
            graph2 = Graph(xlabel=XLABEL, ylabel=YLABEL, x_ticks_minor=5,
                            x_ticks_major=(GRAPH_X_RANGE[1] - GRAPH_X_RANGE[0]) / 15, y_ticks_major=(GRAPH_Y_RANGE[1] - GRAPH_Y_RANGE[0]) / 10,
                            y_grid_label=True, x_grid_label=True, padding=0,
                            x_grid=True, y_grid=True, xmin=GRAPH_X_RANGE[0], xmax=GRAPH_X_RANGE[1], ymin=GRAPH_Y_RANGE[0], ymax=GRAPH_Y_RANGE[1])

        
            new_plot = LinePlot(color= [1, 1, 0, 1])
            new_plot.points = data_graphPlot_class0
            graph2.add_plot(new_plot) 

            new_plot = LinePlot(color= [0, 1, 1, 1])
            new_plot.points = data_graphPlot_class1
            graph2.add_plot(new_plot)  

            self.graph_spaceOne.add_widget(graph2)
        
        
        
    def threadPlotGraph(self,arg,datalist):
        filterAMP = self.sa_filter_inputamp.text

        freq,amp = datalist

        freq = numpy.array(freq)
        amp = numpy.array(amp)

        if filterAMP != "" and filterAMP.isdigit():
            condition = amp > float(filterAMP)
            amp[~condition] = int(filterAMP) + 0.5

        self.plotGraph((freq,amp),filterAMP)
    
    def __threadCooldownText(self,btn,timecount = 2):
        
        btn.disabled = True
        text_default = btn.text
        while True:
            btn.text = "%.2f" % timecount
            time.sleep(0.01)
            timecount -= 0.01

            if timecount <= 0:
                btn.text = text_default
                btn.disabled = False
                break

    def dataset_onbtnstart(self,btn):
        default_text = self.dataset_btnstart.text
        
        datacounttext = self.dataset_inputdatacount.text
        if datacounttext == "":
            return

        if datacounttext.isdigit():
            datacount = int(datacounttext)
        else:
            return
            
        i = 0
        self.variable.status_loop_autodatasetrecord = True
        while self.variable.status_loop_autodatasetrecord:
            self.dataset_btnstart.text = "Recording......."
            threading.Thread(target=self.__threadCooldownText,args=(self.dataset_btnstart,)).start()
            
            freq,amp = self.sound_io.process()
            self.dataset_btnstart.text = default_text
            self.dataset_btnstart.disabled = False

            
            Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

            if self.dataset_switchsavedata.active and self.dataset_inputclassname.text != "" and self.dataset_inputclassname.text.isdigit():
                self.ai.addData({
                "frequency":freq,
                "amplitude":amp,
                "size_frequency":len(freq),
                "size_amplitude":len(amp),
                "classname":int(self.dataset_inputclassname.text)
            })

            self.dataset_textoutput.text = "data : " + str(i + 1)
            self.variable.status_loop_autodatasetrecord = self.dataset_switchsavedata.active


            i += 1

            if i >= datacount:
                self.variable.status_loop_autodatasetrecord = False

        if self.dataset_switchautotrain.active:
            self.filezone_onClickProcessDataset(self.file_btnscandataset)
            self.train_onbtnclick(self.train_btntrain) 
            

    def soundAnalys_onClickRecord(self,btn):
        threading.Thread(target=self.__threadCooldownText,args=(btn,)).start()
        space = 50 if self.sa_filter_inputspace.text == "" or not self.sa_filter_inputspace.text.isdigit() else int(self.sa_filter_inputspace.text)
        self.sound_io.soundOption.FREQ_SPACE = space
        freq,amp = self.sound_io.process()
        


        Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

    def soundAnalys_onClickPlayLastRecord(self,btn):
        threading.Thread(target=self.__threadCooldownText,args=(btn,)).start()
        self.sound_io.playSound()

    def soundAnalys_onClickPlayGenSound(self,btn):
        freq_gen = self.sa_gensound_freq.text
        sec_gen  = self.sa_gensound_sec.text

        if freq_gen == "" or not freq_gen.isdigit() or sec_gen == "" or not freq_gen.isdigit():
            return

        freq = int(freq_gen)
        sec  = int(sec_gen)

        self.sound_io.genSound(freq=freq,durationIn=sec)

    def soundAnalys_onClickSave(self,btn):

        self.ai.addData({
                "frequency":freq,
                "amplitude":amp,
                "size_frequency":len(freq),
                "size_amplitude":len(amp),
                "classname":int(self.dataset_inputclassname.text)
        })

    def filezone_onClickProcessDataset(self,btn):

        train_persen_text = self.file_inputtrain_persen.text
        if train_persen_text == "" or not train_persen_text.isdigit():
            return 

        x_train,y_train,x_val,y_val,x_test,y_test = self.ai.getSplitDataset(self.ai.getDataFileCount(),int(train_persen_text))
        self.variable.DataSet.set(x_train,y_train,x_val,y_val,x_test,y_test)


        if not self.variable.DataSet.isData():
            return
        
        self.variable.DataSet.shape()

        self.file_traincount.text = str(x_train.shape[0])
        self.file_validcount.text = str(x_val.shape[0])
        self.file_testcount.text = str(x_test.shape[0])
        self.file_classcount.text = str(self.ai.dataother.getClassCount())


        
        def setTextInputClassName(dt):
            self.train_inputclasscount.text = str(self.ai.dataother.getClassCount())
        Clock.schedule_once(setTextInputClassName )

    def train_onbtnclick(self,btn):
        
        if not self.variable.DataSet.isData():
            self.train_textoutput.text = 'Dataset is not ready!!!'
            return

        class_count_text = self.train_inputclasscount.text
        class_epoch_text = self.train_inputepoch.text
        if class_count_text == "" and not class_count_text.isdigit():
            self.train_textoutput.text = 'input classcount number now!!'
            return
        if class_epoch_text == "" and not class_epoch_text.isdigit():
            self.train_textoutput.text = 'input epoch number now!!'
            return
        

        
        class_count = int(class_count_text)
        epoch = int(class_epoch_text)
 

        self.train_textoutput.text = 'Training....'

        self.variable.MODEL = self.ai.train(self.variable.DataSet.x_train,self.variable.DataSet.y_train,self.variable.DataSet.x_valid,self.variable.DataSet.y_valid,class_count,epoch)

        accuracy,loss = self.ai.getAccuracy(self.variable.DataSet.x_test,self.variable.DataSet.y_test)
        self.variable.TestAccuracy = accuracy
        self.variable.TestLoss = loss


        self.train_textoutput.text = 'Success'

    def predict_onbtnclick(self,btn):
        
        self.variable.status_loop_autopredict = True
        self.predict_btnpredict.disabled = True
        while self.variable.status_loop_autopredict:
            freq,amp = self.sound_io.process()

            Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

            classname,max_con = self.ai.predict(amplitude=amp)

            self.predict_class.color = self.variable.color_class_list[classname]
            self.predict_maxconfi.color = self.variable.color_class_list[classname]

            self.predict_class.text = str(classname)
            self.predict_maxconfi.text = str('%.2f' % (max_con * 100)) + "%"

            self.variable.status_loop_autopredict = self.predict_swautopredict.active

        self.predict_btnpredict.disabled = False



class ScreenManagerApp(App):
    def build(self):

        Window.size = (1200, 600)
        Window.minimum_width = 800
        Window.minimum_height = 600
        Window.fullscreen = 'auto'

        Builder.load_file('main_screen.kv')
        Builder.load_file('file_manager.kv')

        sm = ScreenManager()
        sm.add_widget(MainScreen(screen_name='MainScreen'))
        sm.add_widget(FileManagerScreen(screen_name='FileManagerScreen'))
        return sm
    
    def changeto_filemanagerScreen(self):
        self.root.current = 'FileManagerScreen'
    
    # def change



sm = ScreenManagerApp()
sm.run()




