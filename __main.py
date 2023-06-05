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




"""
def onRecord(btn,mygrid):

    def threadCooldownText(btn):
        timecount = 2
        while True:
            btn.text = "%.2f" % timecount
            time.sleep(0.01)
            timecount -= 0.01

            if timecount <= 0:
                btn.text = "Record"
                break
    
    datacounttext = mygrid.input_autorec_datacount.text

    if datacounttext == "" and not datacounttext.isdigit():
        datacount = 1
    else:
        datacount = int(datacounttext)

    

    for x in range(datacount):
        btn.text = "Recording......."
        threading.Thread(target=threadCooldownText,args=(btn,)).start()
        btn.disabled = True
        freq,amp = sound_io.process()
        btn.text = "Record"
        btn.disabled = False

        variable.freq = freq
        variable.amp = amp

        
        Clock.schedule_once(lambda argment : threadPlotGraph(argment,mygrid,(freq,amp)), 0)

        if mygrid.swAutoSave.active:
            onSaveData(btn=btn,mygrid=mygrid)

        mygrid.labelstatus.text = "data = " + str(x + 1)

def onPlaySound(btn):

    def playsound_running(btn):

        timecount = sound_io.getOutputDuratoion()
        default_text = btn.text
        btn.disabled = True
        while True:
            btn.text = "%.2f" % timecount

            time.sleep(0.01)
            timecount -= 0.01

            if timecount <= 0:
                btn.text = default_text
                btn.disabled = False
                break
        

    threading.Thread(target=playsound_running,args=(btn,)).start()

    sound_io.playSound()

def onPlayGenSound(btn,freqIn,secIn):
    freq = freqIn.text
    sec = secIn.text

    print(f"freq = {freq}    sec = {sec}")

    freqint = 50 if not freq.isdigit() else int(freq)
    durationInt = 2.0 if not sec.isdigit() else float(sec)
    sound_io.genSound(freq=freqint,durationIn= durationInt)

def onLoadData(btn,mygrid):
    path = easygui.fileopenbox(default=datamanager.DIRECTORY_ROOT_PATH)
    if not path:
        return
    data = datamanager.getData(path=path)

    freq = data['frequency']
    amp = data['amplitude']
    
    Clock.schedule_once(lambda argment : threadPlotGraph(argment,mygrid,(freq,amp)), 0)

"""
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

    sa_btnrecord = ObjectProperty(None)
    sa_btnplaylastsound = ObjectProperty(None)
    sa_gensound_freq = ObjectProperty(None)
    sa_gensound_sec = ObjectProperty(None)
    sa_gensound_btnplay = ObjectProperty(None)
    sa_filter_inputamp = ObjectProperty(None)
    sa_btnsavedata = ObjectProperty(None)
    sa_btnloaddata = ObjectProperty(None)

    train_btntrain = ObjectProperty(None)
    train_inputclasscount = ObjectProperty(None)
    train_textoutput = ObjectProperty(None)
    train_inputepoch = ObjectProperty(None)

    predict_btnpredict = ObjectProperty(None)
    predict_textoutput = ObjectProperty(None)

    model_model = ObjectProperty(None)
    model_accuracy = ObjectProperty(None)

    file_btnscandataset = ObjectProperty(None)
    file_filecount = ObjectProperty(None)
    file_traincount = ObjectProperty(None)
    file_validcount = ObjectProperty(None)
    file_testcount = ObjectProperty(None)
    file_classcount = ObjectProperty(None)
    file_inputtrain_persen = ObjectProperty(None)

    graph_space = ObjectProperty(None)

    graph = None
    graphLayout = []
    graphPoint = []

    

    def __init__(self,screen_name, **kwargs):
        super().__init__(**kwargs)
        self.name = screen_name

        self.variable = Variable()
        self.sound_io = SoundIO()
        self.ai = AI()

        self.dataset_btnstart.bind(on_press=lambda btn: threading.Thread(target=self.dataset_onbtnstart,args=(btn,)).start())  
        self.train_btntrain.bind(on_press = lambda btn : threading.Thread(target=self.train_onbtnclick,args=(btn,)).start())
        self.predict_btnpredict.bind(on_press = lambda btn : threading.Thread(target=self.predict_onbtnclick,args=(btn,)).start())
        self.file_btnscandataset.bind(on_press = lambda btn : threading.Thread(target=self.filezone_onClickProcessDataset,args=(btn,)).start())

        self.sa_btnrecord.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickRecord,args=(btn,)).start())
        self.sa_btnplaylastsound.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickPlayLastRecord,args=(btn,)).start())
        self.sa_gensound_btnplay.bind(on_press = lambda btn : threading.Thread(target=self.soundAnalys_onClickPlayGenSound,args=(btn,)).start())
        
        def onDialog(btn):
            dialog = MyDialog()
            dialog.open()

        # self.sa_btnrecord.bind(on_press=onDialog)


        def update_label(dt):

            self.model_model.text = "Has Model" if self.variable.MODEL != None else "No Model"
            
            if self.variable.Accuracy != None:
                self.model_accuracy.text = str("%.2f" % (self.variable.Accuracy * 100) + "%")

            self.file_filecount.text = str(self.ai.getDataFileCount())

            self.ai.createBaseResource()


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

    def plotGraph(self,data,filterAMP):
        
        DATA_FORMATT = "((1,2),(5,6),...)"

        XLABEL = 'Frequency (Hz)'
        YLABEL = 'Amplitude (db)'
        GRAPH_X_RANGE = (0,3000)
        GRAPH_Y_RANGE = (0,100)
        
        
        self.graph_space.clear_widgets()
        
        if len(self.graphPoint) > 1:
            self.graphPoint.remove(self.graphPoint[0])

        self.graphPoint.append(data)

        time.sleep(0.1)

        self.graph = Graph(xlabel=XLABEL, ylabel=YLABEL, x_ticks_minor=5,
                      x_ticks_major=200, y_ticks_major=10,
                      y_grid_label=True, x_grid_label=True, padding=0,
                      x_grid=True, y_grid=True, xmin=GRAPH_X_RANGE[0], xmax=GRAPH_X_RANGE[1], ymin=GRAPH_Y_RANGE[0], ymax=GRAPH_Y_RANGE[1])
        
        i = 0
        for points_item in self.graphPoint:
            new_plot = LinePlot(color= [1, 0, 0, 1] if i == 0 else [0, 1, 0, 1])
            new_plot.points = points_item
            self.graph.add_plot(new_plot)
            i+=1

        if filterAMP != "" and filterAMP.isdigit():
            new_plot = LinePlot(color= [0, 0, 1, 1])
            new_plot.points = [(0,int(filterAMP)),(3000,int(filterAMP))]
            self.graph.add_plot(new_plot)
        
        self.graph_space.add_widget(self.graph)
        self.graphLayout.append(self.graph)
        
    def threadPlotGraph(self,arg,datalist):
        filterAMP = self.sa_filter_inputamp.text

        freq,amp = datalist

        if filterAMP != "" and filterAMP.isdigit():
            condition = amp > float(filterAMP)
            amp[~condition] = int(filterAMP) + 0.5

        data_graphPlot = list(zip(freq, amp))

        self.plotGraph(data_graphPlot,filterAMP)
    
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
            

        for x in range(datacount):
            self.dataset_btnstart.text = "Recording......."
            threading.Thread(target=self.__threadCooldownText,args=(self.dataset_btnstart,)).start()
            
            freq,amp = self.sound_io.process()
            self.dataset_btnstart.text = default_text
            self.dataset_btnstart.disabled = False

            
            Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

            if self.dataset_switchsavedata.active and self.dataset_inputclassname.text != "" and self.dataset_inputclassname.text.isdigit():
                self.ai.addData({
                "frequency":freq.tolist(),
                "amplitude":amp.tolist(),
                "size_frequency":len(freq),
                "size_amplitude":len(amp),
                "classname":int(self.dataset_inputclassname.text)
            })

            self.dataset_textoutput.text = "data : " + str(x + 1)

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
 

        self.train_textoutput.text = 'Ok, Training....'

        self.variable.MODEL = self.ai.train(self.variable.DataSet.x_train,self.variable.DataSet.y_train,class_count,epoch)

        accuracy,loss = self.ai.getAccuracy(self.variable.DataSet.x_test,self.variable.DataSet.y_test)
        self.variable.Accuracy = accuracy


        self.train_textoutput.text = 'Success, Train Model'

    def predict_onbtnclick(self,btn):
        freq,amp = self.sound_io.process()

        Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

        classname,confidence = self.ai.predict(amplitude=amp)

        suretext  = "unsure" if confidence < 0.995 else "i'm sure"
        text = f"classname : {classname}\nconfidence = {'%.2f' % (confidence * 100)}%\n{suretext}"

        print(text)
        self.predict_textoutput.text = text

    
    def soundAnalys_onClickRecord(self,btn):
        threading.Thread(target=self.__threadCooldownText,args=(btn,)).start()
        freq,amp = self.sound_io.process()

        Clock.schedule_once(lambda argment : self.threadPlotGraph(argment,(freq,amp)), 0)

    def soundAnalys_onClickPlayLastRecord(self,btn):
        print("on soundAnalys_onClickPlayLastRecord")
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

    def filezone_onClickProcessDataset(self,btn):

        train_persen_text = self.file_inputtrain_persen.text
        if train_persen_text == "" or not train_persen_text.isdigit():
            return

        x_train,y_train,x_val,y_val,x_test,y_test,classname_count = self.ai.getSplitDataset(self.ai.getDataFileCount(),int(train_persen_text))
        self.variable.DataSet.set(x_train,y_train,x_val,y_val,x_test,y_test)

        if not self.variable.DataSet.isData():
            return

        self.file_traincount.text = str(x_train.shape[0])
        self.file_validcount.text = str(x_val.shape[0])
        self.file_testcount.text = str(x_test.shape[0])
        self.file_classcount.text = str(len(classname_count))


        
        def setTextInputClassName(dt):
            self.train_inputclasscount.text = str(classname_count)
        Clock.schedule_once(setTextInputClassName )




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
















"""
dataName = 0

def onRecord(ui : UI):
    global datamanager,dataName
    frames = sound_io.record()

    ui.printStatus("Record, Finish")
    ui.btnRec.config(state="normal")


    freq,ampli = sound_io.GetFFT(frames)


    classname = ui.getClassname()
    if classname.isdigit():
        datamanager.addData(str(dataName),{
            'x':ampli,
            'classname': int(classname)
        })

        dataName = dataName + 1

    # freq_filter,ampli_filter = sound_io.filter(freq,ampli)

    ui.plot_graph(x=freq,y=ampli,x_label='Frequency (Hz)',y_label='Amplitude')

def onTrainClick(ui : UI,trainclass : ClassificationModel,data : DataManager):
    print("onTrainClick")
    
    print(f"data list len = {data.getDataCount()}")

    X_Train,Y_Train = data.getDataForTrain()

    trainclass.train(x_train=X_Train,y_train=Y_Train,epochsIn=100)

    

def onPredictClick(ui : UI,trainclass : ClassificationModel,data : DataManager):
    
    ui.printStatus("predict record....")
    frames = sound_io.record()
    freq,amp = sound_io.GetFFT(frames=frames)


    x_test = numpy.array(amp)

"""
















# ui = UI()

# ui.btnRec.configure(command=lambda:threading.Thread(target=onRecord,args=(ui,)).start())
# ui.btnTrain.configure(command=lambda:threading.Thread(target=onTrainClick,args=(ui,classification,datamanager,)).start())
# ui.btnPredict.configure(command=lambda:threading.Thread(target=onPredictClick,args=(ui,classification,datamanager,)).start())
# ui.mainloop() 


