import kivy.garden.graph
import numpy

class DataObject:
    def __init__(self) -> None:
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test  = None
        self.y_test  = None

    def set(self,x_train = None,y_train = None,x_valid = None,y_valid = None,x_test = None,y_test = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test  = x_test
        self.y_test  = y_test

    def shape(self):
        print(f"x_train = {self.x_train.shape}")
        print(f"y_train = {self.y_train.shape}")
        print(f"x_valid = {self.x_valid.shape}")
        print(f"y_valid = {self.y_valid.shape}")
        print(f"x_test = {self.x_test.shape}")
        print(f"y_test = {self.y_test.shape}")

    def isTrain(self):
        return (numpy.any(self.x_train) and numpy.any(self.y_train))
    
    def isValid(self):
        return (numpy.any(self.x_valid) and numpy.any(self.y_valid))
    
    def isTest(self):
        return (numpy.any(self.x_test) and numpy.any(self.y_valid))
    
    def isData(self):
        return (self.isTrain() and self.isValid() and self.isTest())

class GraphHistory:

    def __init__(self) -> None:
        self.__MAX_HIS = 1
        self.GRAPH_COLOR = [
            # [1,0,0,1],
            [1,0,0,1],
            [0,1,0,1]
        ]

        self.__graph_point = []


        self.graphOneAvg = {
            "class_0":None,
            "class_1":None,
            "now":None
        }

    def addPoint(self,pointIn):
        if len(self.__graph_point) > self.__MAX_HIS:
            self.__graph_point.remove(self.__graph_point[0])
        
        self.__graph_point.append(pointIn)

    def getCount(self):
        return len(self.__graph_point)
    
    def getGraphPoint(self):
        return self.__graph_point

    


class Variable:
    def __init__(self) -> None:
        self.MODEL = None
        self.TestAccuracy = None
        self.TestLoss = None

        self.DataSet : DataObject = DataObject()

        self.color_class_list = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]

        self.graphHistory = GraphHistory()

        self.status_loop_autopredict = False
        self.status_loop_autodatasetrecord = False

        self.datafilecount = 0