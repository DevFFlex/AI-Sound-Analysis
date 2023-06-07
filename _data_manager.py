import os
import sys
import pandas as pd
import json
import numpy
import threading
import time
import datetime
from sklearn.model_selection import train_test_split
    

class DataManager:

    def __init__(self,app):
        self.__app = app

        self.DATA_FOLDERNAME = 'dataset'

        self.MODEL_FILENAME = 'modelfile.h5'
        self.INFO_FILENAME = "info.json"

        self.DIRECTORY_FOLDER_PATH = os.getcwd()

        self.DIRECTORY_DATASET_PATH = os.path.join(self.DIRECTORY_FOLDER_PATH,self.DATA_FOLDERNAME)

        self.INFO_FILE_PATH = os.path.join(self.DIRECTORY_DATASET_PATH,self.INFO_FILENAME)

        self.DATA_CLASSCOUNT = 0
        self.DATA_FILECOUNT = 0
        self.DATA_CLASS = {
            'Class0':0,
            'Class1':0,
        }

        self.createBaseResource()  
        self.__filecountnow = self.__getDataFileCount() 


        print("constructor dataManager")

        t1 = threading.Thread(target=self.__filescaner) 
        t1.setName('FileScanner')
        t1.start()


    def __filescaner(self):
        self.DATA_CLASSCOUNT = self.__getDataClassCount()
        self.__app.avgGraphUpdate()

        while True:
            
            try:
                self.createBaseResource()
            
                self.DATA_FILECOUNT = self.__getDataFileCount()

                if self.__filecountnow != self.DATA_FILECOUNT:
                    self.__filecountnow = self.DATA_FILECOUNT
                    if self.DATA_FILECOUNT != 0:
                        self.__app.avgGraphUpdate()
                    
                    self.DATA_CLASSCOUNT = self.__getDataClassCount()
            except:
                pass
            
            time.sleep(1)

    def createBaseResource(self):
        if not self.isFolder(self.DIRECTORY_DATASET_PATH):
            os.mkdir(self.DIRECTORY_DATASET_PATH)

        if not self.isFile(self.INFO_FILE_PATH):
            with open(self.INFO_FILE_PATH,"w+") as f:
                pass

    def addData(self,new_data):
        filename = datetime.datetime.now().strftime("%H_%M_%S_%f_%d_%m_%Y") + "_" + str(new_data['classname'])
        
        with open(os.path.join(self.DIRECTORY_DATASET_PATH,filename+".json"), "w") as file:
            json.dump(new_data, file)
    
    def __getDataFileCount(self):
        if self.isFolder(self.DIRECTORY_DATASET_PATH):
            return len(os.listdir(self.DIRECTORY_DATASET_PATH)) - 1
        else:
            return 0
    
    def __getDataClassCount(self):
        classnamecountlist = []

        class0list = []
        class1list = []

        for filename in os.listdir(self.DIRECTORY_DATASET_PATH):
            if filename.endswith('.json'):
                if filename == self.INFO_FILENAME:
                    continue
                file_path = os.path.join(self.DIRECTORY_DATASET_PATH, filename)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    amplitude = data['amplitude']
                    classname = data['classname']

                    if classname == 0:
                        class0list.append(True)
                    elif classname == 1:
                        class1list.append(True)

        

                    classnamecountlist.append(classname)

        self.DATA_CLASS['Class0'] = len(class0list)
        self.DATA_CLASS['Class1'] = len(class1list)

        class_count = len(list(set(classnamecountlist)))

        del class0list
        del class1list
        del classnamecountlist

        return class_count


    def getData(self,path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
    
    def getSplitDataset(self,train_persen = None):
        return_null = [None,None,None,None,None,None]
        if not train_persen:
            return return_null

        x_trainlist = []
        y_trainlist = []

        keylists = []
        

        for filename in os.listdir(self.DIRECTORY_DATASET_PATH):
            if filename.endswith('.json'):
                if filename == self.INFO_FILENAME:
                    continue
                file_path = os.path.join(self.DIRECTORY_DATASET_PATH, filename)
                
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    amplitude = data['amplitude']
                    classname = data['classname']

                    for key in data.keys():
                        keylists.append(key)


                    
                    x_trainlist.append(amplitude)
                    y_trainlist.append(classname)

                    

        
        keylists_count = list(set(keylists))


        X = numpy.array(x_trainlist)
        Y = numpy.array(y_trainlist)



        test_persen_use = 100 - train_persen

        data_count_cal = self.DATA_FILECOUNT
        data_count_cal -= ((100 - train_persen) / 100) * self.DATA_FILECOUNT
        data_count_cal -= ((100 - train_persen) / 100) * data_count_cal


        if data_count_cal < 1:
            return return_null


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_persen_use / 100, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_persen_use / 100, random_state=1)

        #normalize
        # X_train /= 1000
        # X_val /= 1000
        # X_test /= 1000

        return [X_train,y_train,X_val,y_val,X_test,y_test]

    def getAverageClass(self):
        try:
            self.DATA_FILECOUNT = self.__getDataClassCount()

            class0list = []
            class1list = []
            
            isadd = True

            freq = None

            for filename in os.listdir(self.DIRECTORY_DATASET_PATH):
                if filename.endswith('.json'):
                    if filename == self.INFO_FILENAME:
                        continue
                    file_path = os.path.join(self.DIRECTORY_DATASET_PATH, filename)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        amplitude = data['amplitude']
                        classname = data['classname']

                        if isadd:
                            isadd = False
                            freq = data['frequency']


                        if classname == 0:
                            class0list.append(amplitude)
                        elif classname == 1:
                            class1list.append(amplitude)


            # class0numpy = numpy.array(class0list)
            # class1numpy = numpy.array(class1list)


            class0mean = numpy.mean(class0list,axis=0) if len(class0list) > 0 else None
            class1mean = numpy.mean(class1list,axis=0) if len(class1list) > 0 else None

            return class0mean,class1mean,freq
        except:
            return None,None,None

    def isFile(self,path):
        return os.path.isfile(path)

    def isFolder(self,path):
        return os.path.isdir(path)
    

