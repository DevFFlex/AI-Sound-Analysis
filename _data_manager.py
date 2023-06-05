import os
import sys
import pandas as pd
import json
import numpy
import time
import datetime
from sklearn.model_selection import train_test_split

class DataManager:

    def __init__(self):
        self.DATA_FOLDERNAME = 'dataset'

        self.MODEL_FILENAME = 'modelfile.h5'
        self.INFO_FILENAME = "info.json"

        self.DIRECTORY_FOLDER_PATH = os.getcwd()

        self.DIRECTORY_DATASET_PATH = os.path.join(self.DIRECTORY_FOLDER_PATH,self.DATA_FOLDERNAME)

        self.INFO_FILE_PATH = os.path.join(self.DIRECTORY_DATASET_PATH,self.INFO_FILENAME)


        self.createBaseResource()    
    

    def createBaseResource(self):
        if not self.isFolder(self.DIRECTORY_DATASET_PATH):
            os.mkdir(self.DIRECTORY_DATASET_PATH)

        if not self.isFile(self.INFO_FILE_PATH):
            with open(self.INFO_FILE_PATH,"w+") as f:
                pass

    def addData(self,new_data):
        filename = datetime.datetime.now().strftime("%H_%M_%S_%f_%d_%m_%Y")
        
        with open(os.path.join(self.DIRECTORY_DATASET_PATH,filename+".json"), "w") as file:
            json.dump(new_data, file)
    
    def getDataFileCount(self):
        if self.isFolder(self.DIRECTORY_DATASET_PATH):
            return len(os.listdir(self.DIRECTORY_DATASET_PATH)) - 1
        else:
            return -1
    
    def getData(self,path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
    
    def getSplitDataset(self,data_count,train_persen = None):

        if not train_persen:
            return None,None,None,None,None,None,None

        x_trainlist = []
        y_trainlist = []

        keylists = []
        classnamecountlist = []

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

                    classnamecountlist.append(classname)
        

        classname_count = list(set(classnamecountlist))


        keylists_filter = list(set(keylists))
        print(keylists_filter)

        print(f"train {len(os.listdir(self.DIRECTORY_DATASET_PATH)) - 1} data finish")

        X = numpy.array(x_trainlist)
        Y = numpy.array(y_trainlist)

        train_persen_use = train_persen
        test_persen_use = 100 - train_persen

        data_count_cal = data_count
        data_count_cal -= ((100 - train_persen) / 100) * data_count
        data_count_cal -= ((100 - train_persen) / 100) * data_count_cal

        if data_count_cal < 1:
            return None,None,None,None,None,None,None

        # 10/100 * 20

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_persen_use / 100, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_persen_use / 100, random_state=1)
        print(f'#traning = {X_train.shape[0]}')
        print(f'#validation = {X_val.shape[0]}')
        print(f'#test = {X_test.shape[0]}')


        return (X_train,y_train,X_val,y_val,X_test,y_test,classname_count)

    def isFile(self,path):
        return os.path.isfile(path)

    def isFolder(self,path):
        return os.path.isdir(path)
    

