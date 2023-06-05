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

    def isTrain(self):
        return (numpy.any(self.x_train) and numpy.any(self.y_train))
    
    def isValid(self):
        return (numpy.any(self.x_valid) and numpy.any(self.y_valid))
    
    def isTest(self):
        return (numpy.any(self.x_test) and numpy.any(self.y_valid))
    
    def isData(self):
        return (self.isTrain() and self.isValid() and self.isTest())

class Variable:
    def __init__(self) -> None:
        self.MODEL = None
        self.Accuracy = None

        self.DataSet : DataObject = DataObject()