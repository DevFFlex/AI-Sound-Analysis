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

from kivy.properties import ObjectProperty
from kivy.garden.graph import Graph, MeshLinePlot,LinePlot

from kivy.base import runTouchApp
from kivy.graphics import Color, Line
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from file.ex_class.GraphManage import GraphManage
from file.ex_class.widgetAction import ButtonCoolDown, InputTextWidget
from file._sound_io import SoundIO
from file._variable import Variable
from file._ai_class import AI

class MainScreen(Screen):
    dataset_inputclassname = ObjectProperty(None)
    dataset_textoutput = ObjectProperty(None)
    dataset_btnstart = ObjectProperty(None)
    dataset_inputdatacount = ObjectProperty(None)
    dataset_switchsavedata = ObjectProperty(None)
    dataset_switchautotrain = ObjectProperty(None)

    sa_btnrecord = ObjectProperty(None)
    sa_btnopenfile = ObjectProperty(None)
    sa_btnplaylastsound = ObjectProperty(None)
    sa_gensound_freq = ObjectProperty(None)
    sa_gensound_sec = ObjectProperty(None)
    sa_gensound_btnplay = ObjectProperty(None)
    sa_filter_inputamp = ObjectProperty(None)
    sa_btnsavedata = ObjectProperty(None)
    sa_btnloaddata = ObjectProperty(None)
    sa_filter_inputspace = ObjectProperty(None)

    train_btntrain = ObjectProperty(None)
    train_textoutput = ObjectProperty(None)
    train_inputepoch = ObjectProperty(None)
    train_btntestmodel = ObjectProperty(None)
    train_btnsavemodel = ObjectProperty(None)

    predict_btnpredict = ObjectProperty(None)
    predict_class = ObjectProperty(None)
    predict_maxconfi = ObjectProperty(None)
    predict_swautopredict = ObjectProperty(None)
    predict_btnpredictmodel_ = ObjectProperty(None)

    model_model = ObjectProperty(None)
    model_testaccuracy = ObjectProperty(None)
    model_testloss = ObjectProperty(None)

    # file_btnscandataset = ObjectProperty(None)
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
    graph_labelclasszero = ObjectProperty(None)
    graph_labelclassone = ObjectProperty(None)

    output = ObjectProperty(None)

    def onClickThread(self,btn,callback):
        btn.bind(on_press = lambda btn : threading.Thread(target=callback,args=(btn,)).start())

    def onTrainingCallbackFunction(self, epoch, logs):
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        self.train_textoutput.text = f"epoch {epoch} | Accuracy = {'%.2f' % accuracy} | loss = {'%.2f' % loss}"

    def __init__(self,screen_name, **kwargs):
        super().__init__(**kwargs)
        self.name = screen_name

        self.variable = Variable()
        self.sound_io = SoundIO()
        self.ai = AI(self,self.variable)

        self.graph1 = GraphManage(self.graph_space,self.variable)
        self.graph1.setOption(self.graph_inputMinScaleX,self.graph_inputMaxScaleX,self.graph_inputMinScaleY,self.graph_inputMaxScaleY)
        self.graph1.plotGraph()

        self.onClickThread(self.dataset_btnstart,self.dataset_onbtnstart)
        self.onClickThread(self.train_btntrain,self.onClick_Train)
        self.onClickThread(self.train_btnsavemodel,self.onClick_SaveModel)
        self.onClickThread(self.train_btntestmodel,self.onClick_TestModel)
        self.onClickThread(self.predict_btnpredict,self.onClick_Predict)
        self.onClickThread(self.predict_btnpredictmodel_,self.onClick_PredictClassification)
        self.onClickThread(self.file_btnopenfolder,lambda btn : os.system(f"start {os.getcwd()}"))
        self.onClickThread(self.sa_btnrecord,self.soundAnalys_onClickRecord)
        self.onClickThread(self.sa_btnopenfile,self.soundAnalys_onClickOpenFile)
        self.onClickThread(self.sa_btnplaylastsound,self.soundAnalys_onClickPlayLastRecord)
        self.onClickThread(self.sa_gensound_btnplay,self.soundAnalys_onClickPlayGenSound)


        self.file_inputtrain_persen.bind(text = self.onType_ProcessFile)

        def update_label(dt):
            
            self.model_model.color = [0,1,0,1] if self.variable.MODEL != None else [1,0,0,1]
            self.model_model.text = "Model Is Ready" if self.variable.MODEL != None else "No Model"
            
            if self.variable.TestAccuracy != None and self.variable.TestLoss != None:
                self.model_testaccuracy.text = str("%.2f" % (self.variable.TestAccuracy * 100) + "%")
                self.model_testloss.text = str("%.2f" % (self.variable.TestLoss * 100) + "%")

            self.file_filecount.text = str(self.ai.DATA_FILECOUNT)

            if self.variable.DataSet.isData():
                self.train_btntrain.disabled = False
                self.train_btntestmodel.disabled = False
                self.train_btnsavemodel.disabled = False
            else:
                self.train_btntrain.disabled = True
                self.train_btntestmodel.disabled = True
                self.train_btnsavemodel.disabled = True


            if self.ai.isModel():
                self.predict_btnpredict.disabled = False
            else:
                self.predict_btnpredict.disabled = True

            self.graph_labelclasszero.text = str(self.ai.DATA_CLASS['Class0'])
            self.graph_labelclassone.text  = str(self.ai.DATA_CLASS['Class1'])
        Clock.schedule_interval(lambda dt: update_label(dt,), 1)

   
    def plotGraph(self,frequency,amplitude):

        def t_plot():

            def function(arg,freqamp,filterAMP):
                freq,amp = freqamp
                data = list(zip(freq, amp))
                

                self.variable.graphHistory.graphInputOption(self.graph_inputMinScaleX,self.graph_inputMaxScaleX,self.graph_inputMinScaleY,self.graph_inputMaxScaleY)

                self.graph_inputMinScaleX.text = str(self.variable.graphHistory.GRAPH_X_RANGE[0])
                self.graph_inputMaxScaleX.text = str(self.variable.graphHistory.GRAPH_X_RANGE[1])
                self.graph_inputMinScaleY.text = str(self.variable.graphHistory.GRAPH_Y_RANGE[0])
                self.graph_inputMaxScaleY.text = str(self.variable.graphHistory.GRAPH_Y_RANGE[1])

    
                
                self.graph_space.clear_widgets()
                
                self.variable.graphHistory.addPoint(data)

                self.graph = Graph(xlabel=self.variable.graphHistory.XLABEL, ylabel=self.variable.graphHistory.YLABEL, x_ticks_minor=5,
                            x_ticks_major=(self.variable.graphHistory.GRAPH_X_RANGE[1] - self.variable.graphHistory.GRAPH_X_RANGE[0]) / 15, y_ticks_major=(self.variable.graphHistory.GRAPH_Y_RANGE[1] - self.variable.graphHistory.GRAPH_Y_RANGE[0]) / 10,
                            y_grid_label=True, x_grid_label=True, padding=0,
                            x_grid=True, y_grid=True, xmin=self.variable.graphHistory.GRAPH_X_RANGE[0], xmax=self.variable.graphHistory.GRAPH_X_RANGE[1], ymin=self.variable.graphHistory.GRAPH_Y_RANGE[0], ymax=self.variable.graphHistory.GRAPH_Y_RANGE[1])
                
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

            filterAMP = self.sa_filter_inputamp.text

            freq = numpy.array(frequency)
            amp = numpy.array(amplitude)

            if filterAMP != "" and filterAMP.isdigit():
                condition = amp > float(filterAMP)
                amp[~condition] = int(filterAMP) + 0.5

            Clock.schedule_once(lambda argment : function(argment,(freq,amp),filterAMP), 0)

        t1 = threading.Thread(target=t_plot)
        t1.start()

    def avgGraphUpdate(self):
      
        def t_function():
            def function(arg):
                self.variable.datafilecount = self.ai.DATA_FILECOUNT
                class0,class1,freq = self.ai.getAverageClass()

                if self.ai.DATA_CLASSCOUNT <= 0:
                    return

                self.graph_spaceOne.clear_widgets()

                self.variable.graphHistory.graphInputOption(self.graph_inputMinScaleX,self.graph_inputMaxScaleX,self.graph_inputMinScaleY,self.graph_inputMaxScaleY)

                graph2 = Graph(xlabel=self.variable.graphHistory.XLABEL, ylabel=self.variable.graphHistory.YLABEL, x_ticks_minor=5,
                        x_ticks_major=(self.variable.graphHistory.GRAPH_X_RANGE[1] - self.variable.graphHistory.GRAPH_X_RANGE[0]) / 15, y_ticks_major=(self.variable.graphHistory.GRAPH_Y_RANGE[1] - self.variable.graphHistory.GRAPH_Y_RANGE[0]) / 10,
                        y_grid_label=True, x_grid_label=True, padding=0,
                        x_grid=True, y_grid=True, xmin=self.variable.graphHistory.GRAPH_X_RANGE[0], xmax=self.variable.graphHistory.GRAPH_X_RANGE[1], ymin=self.variable.graphHistory.GRAPH_Y_RANGE[0], ymax=self.variable.graphHistory.GRAPH_Y_RANGE[1])
            

                if numpy.any(class0):
                    data_graphPlot_class0 = list(zip(freq, class0))
                    new_plot = LinePlot(color= [1, 1, 0, 1])
                    new_plot.points = data_graphPlot_class0
                    graph2.add_plot(new_plot) 

                if numpy.any(class1):
                    data_graphPlot_class1 = list(zip(freq, class1))
                    new_plot = LinePlot(color= [0, 1, 1, 1])
                    new_plot.points = data_graphPlot_class1
                    graph2.add_plot(new_plot)  

                self.graph_spaceOne.add_widget(graph2)
            
            Clock.schedule_once(lambda argment : function(argment), 0)
        threading.Thread(target=t_function).start()

    def dataset_onbtnstart(self,btn):

        datacount = InputTextWidget(self.dataset_inputdatacount).input()
        if not datacount:
            return


        
        i = 0
        self.variable.status_loop_autodatasetrecord = True
        
        ButtonCoolDown(btn=self.dataset_btnstart,count = datacount)
        while self.variable.status_loop_autodatasetrecord and self.variable.app_running:       
            
            freq,amp = self.sound_io.process()

            print(type(freq))
            print(type(amp))
            
            self.plotGraph(freq,amp)


            if self.dataset_switchsavedata.active and self.dataset_inputclassname.text != "" and self.dataset_inputclassname.text.isdigit():
                self.ai.addData({
                "frequency":freq.tolist(),
                "amplitude":amp.tolist(),
                "size_frequency":len(freq),
                "size_amplitude":len(amp),
                "classname":int(self.dataset_inputclassname.text)
            })

            self.dataset_textoutput.text = "data : " + str(i + 1)
            # self.variable.status_loop_autodatasetrecord = self.dataset_switchsavedata.active


            i += 1

            if i >= datacount:
                self.variable.status_loop_autodatasetrecord = False
        

        if self.dataset_switchautotrain.active and self.variable.DataSet.isData():
            self.ProcessFile()
            self.Train()
            
    def soundAnalys_onClickRecord(self,btn):
        
        space = 50 if self.sa_filter_inputspace.text == "" or not self.sa_filter_inputspace.text.isdigit() else int(self.sa_filter_inputspace.text)
        self.sound_io.soundOption.FREQ_SPACE = space
        freq,amp = self.sound_io.process()
        
        self.plotGraph(freq,amp)

    def soundAnalys_onClickOpenFile(self,btn):

        path_wavfile = easygui.fileopenbox()
        print(path_wavfile)

        freq,amp = self.sound_io.GetFFTWithSoundFile(path_wavfile)
        print(freq)

    def soundAnalys_onClickPlayLastRecord(self,btn):
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
        pass
        # self.ai.addData({
        #         "frequency":freq,
        #         "amplitude":amp,
        #         "size_frequency":len(freq),
        #         "size_amplitude":len(amp),
        #         "classname":int(self.dataset_inputclassname.text)
        # })

    def ProcessFile(self,train_persen = 80):

        def d(dt):
            self.file_inputtrain_persen.text = str(train_persen)
        Clock.schedule_once(d, 0)

        x_train,y_train,x_val,y_val,x_test,y_test = self.ai.getSplitDataset(train_persen)
        
        print(x_train.shape)
        print(y_train.shape)

        self.variable.DataSet.set(x_train,y_train,x_val,y_val,x_test,y_test)

        if not self.variable.DataSet.isData():
            return
        

        self.file_traincount.text = str(x_train.shape[0])
        self.file_validcount.text = str(x_val.shape[0])
        self.file_testcount.text = str(x_test.shape[0])
        self.file_classcount.text = str(self.ai.DATA_CLASSCOUNT)

    def Train(self,epoch = 50):
        self.train_textoutput.text = 'Training....'


        self.variable.MODEL = self.ai.train(self.variable.DataSet.x_train,self.variable.DataSet.y_train,self.variable.DataSet.x_valid,self.variable.DataSet.y_valid,self.ai.DATA_CLASSCOUNT,epoch,self.onTrainingCallbackFunction)

        accuracy,loss = self.ai.getAccuracy(self.variable.DataSet.x_test,self.variable.DataSet.y_test)
        self.variable.TestAccuracy = accuracy
        self.variable.TestLoss = loss


        self.train_textoutput.text = 'Success'  

    def onType_ProcessFile(self, instance, value):
        if value == "" or not value.isdigit():
            return
        print("Text changed:", value)

        train_persen = int(value)

        self.ProcessFile(train_persen)  
        
    def onClick_Train(self,btn):
        
        if not self.variable.DataSet.isData():
            self.train_textoutput.text = 'Dataset is not ready!!!'
            return
        
        epoch = InputTextWidget(self.train_inputepoch).input()
        if not epoch:
            return
        
        self.Train(epoch)
 
    def onClick_TestModel(self):
        accuracy,loss = self.ai.getAccuracy(self.variable.DataSet.x_test,self.variable.DataSet.y_test)
        self.variable.TestAccuracy = accuracy
        self.variable.TestLoss = loss
        
    def onClick_Predict(self,btn):
        
        self.variable.status_loop_autopredict = True
        self.predict_btnpredict.disabled = True
        while self.variable.status_loop_autopredict and self.variable.app_running:
            freq,amp = self.sound_io.process()

            self.plotGraph(freq,amp)

            classname,max_con = self.ai.predict(amplitude=amp)

            self.predict_class.color = self.variable.color_class_list[classname]
            self.predict_maxconfi.color = self.variable.color_class_list[classname]

            self.predict_class.text = str(classname)
            self.predict_maxconfi.text = str('%.2f' % (max_con * 100)) + "%"
            # self.predict_maxconfi.text = "100%"

            self.variable.status_loop_autopredict = self.predict_swautopredict.active

        self.predict_btnpredict.disabled = False

    def onClick_PredictClassification(self,btn):
        
        btn.disabled = True

        freq,amp = self.sound_io.process()

        self.plotGraph(freq,amp)
        
        amp_re = amp.reshape(1, -1)
        print(amp_re.shape)
        print(amp_re.ndim)
        print(amp_re)
        classname,confidence = self.ai.predict_LogisticRegression(amp_re)

        self.predict_class.color = self.variable.color_class_list[classname[0]]
        self.predict_maxconfi.color = self.variable.color_class_list[classname[0]]

        self.predict_class.text = str(classname)
        # self.predict_maxconfi.text = str("%.2f" % confidence[0][0]) + "\n" + str("%.2f" % confidence[0][1])
        self.predict_maxconfi.text = "100%"
        

        btn.disabled = False

    def onClick_SaveModel(self,btn):
        self.ai.saveModel()