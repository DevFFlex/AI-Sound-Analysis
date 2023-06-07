import threading
import numpy
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot,LinePlot
from _variable import Variable




class GraphManage:


    def __init__(self,space_field,variable) -> None:
        self.__space_field = space_field

        self.__variable : Variable = variable

        self.__XLABEL = 'X'
        self.__YLABEL = 'Y'
        self.__GRAPH_X_RANGE = [0,3000]
        self.__GRAPH_Y_RANGE = [0,100]

        self.__pointlist = []
        self.__colorlinelist = [
             # [1,0,0,1],
            [1,0,0,1],
            [0,1,0,1]
        ]

        self.setOption()


    def isReady(self):
        condition1 = True if self.__gInputMinSX != None and self.__gInputMaxSX != None and self.__gInputMinSY != None and self.__gInputMaxSY != None else False
        condition2 = True if self.__XLABEL != None and self.__XLABEL != None else False
        return condition1 and condition2
        
        
    def setOption(self,g_inputMinSX = None,g_InputMaxSX = None,g_InputMinSY = None,g_InputMaxSY = None):
        self.__gInputMinSX = g_inputMinSX
        self.__gInputMaxSX = g_InputMaxSX
        self.__gInputMinSY = g_InputMinSY
        self.__gInputMaxSY = g_InputMaxSY


    def setLabel(self,x_label,y_label):
        self.__XLABEL = x_label
        self.__YLABEL = y_label

    def __graphInputOption(self):
        xminrange_text = self.__gInputMinSX.text
        xmaxrange_text = self.__gInputMaxSX.text
        yminrange_text = self.__gInputMinSY.text
        ymaxrange_text = self.__gInputMaxSY.text

        if xminrange_text != "" and xminrange_text.isdigit():
            self.__GRAPH_X_RANGE[0] = int(xminrange_text)
        
        if xmaxrange_text != "" and xmaxrange_text.isdigit():
            self.__GRAPH_X_RANGE[1] = int(xmaxrange_text)
        
        if yminrange_text != "" and yminrange_text.isdigit():
            self.__GRAPH_Y_RANGE[0] = int(yminrange_text)
        
        if ymaxrange_text != "" and ymaxrange_text.isdigit():
            self.__GRAPH_Y_RANGE[1] = int(ymaxrange_text)

        self.__gInputMinSX.text = str(self.__GRAPH_X_RANGE[0])
        self.__gInputMaxSX.text = str(self.__GRAPH_X_RANGE[1])
        self.__gInputMinSY.text = str(self.__GRAPH_Y_RANGE[0])
        self.__gInputMaxSY.text = str(self.__GRAPH_Y_RANGE[1])

    def __t_plot(self,x : numpy.ndarray, y :numpy.ndarray):

        def function(arg,x,y):

            self.__graphInputOption()

            points = list(zip(x, y))
            
            self.__space_field.clear_widgets()
            
            self.__pointlist.append(points)

            graph = Graph(xlabel=self.__XLABEL, ylabel=self.__YLABEL, x_ticks_minor=5,
                        x_ticks_major=(self.__GRAPH_X_RANGE[1] - self.__GRAPH_X_RANGE[0]) / 15, y_ticks_major=(self.__GRAPH_Y_RANGE[1] - self.__GRAPH_Y_RANGE[0]) / 10,
                        y_grid_label=True, x_grid_label=True, padding=0,
                        x_grid=True, y_grid=True, xmin=self.__GRAPH_X_RANGE[0], xmax=self.__GRAPH_X_RANGE[1], ymin=self.__GRAPH_Y_RANGE[0], ymax=self.__GRAPH_Y_RANGE[1])
            
            i = 0
            for point in self.__pointlist:
                new_plot = LinePlot(color= self.__colorlinelist[i])
                new_plot.points = point
                graph.add_plot(new_plot)
                i+=1
            
            self.__space_field.add_widget(graph)

        Clock.schedule_once(lambda argment : function(argment,x,y), 0)

    def plotGraph(self,x : numpy.ndarray = numpy.array([0,0]),y : numpy.ndarray = numpy.array([0,0])):
        
        if self.isReady():
            t1 = threading.Thread(target=self.__t_plot,args=(x,y,))
            t1.start()
        else:
            print('graph not ready')



    
        