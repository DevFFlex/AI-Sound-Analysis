
import pandas as pd
import numpy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Dropout
import soundfile as sf
import librosa
import os
import json
import easygui
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import ast
import time
import random
from file._data_manager import DataManager

class AI(DataManager):

    def __init__(self,app,variable):
        super().__init__(app,variable)
        self.__model = None
        
    def isModel(self):
        return self.__model         


    def train(self,x_train,y_train,x_val,y_val,class_count,epochsIn = 50,trainCallbackFunction = None):
        self.__classcount = class_count

        self.__model = None
        self.__model = keras.models.Sequential()
 
        self.__model.add(keras.layers.Dense(119, activation='relu', input_shape=(x_train.shape[1],)))
        self.__model.add(Dropout(0.2))
        self.__model.add(keras.layers.Dense(100, activation='relu'))
        self.__model.add(Dropout(0.2))
        self.__model.add(keras.layers.Dense(self.__classcount, activation='sigmoid'))

        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # ตั้งค่าฟังก์ชันสูญเสียและตัวปรับแบบเหมาะสม

        y_train_categorical = keras.utils.to_categorical(y_train, self.__classcount)
        y_valid_categorical = keras.utils.to_categorical(y_val,self.__classcount)

        callfunc = [keras.callbacks.LambdaCallback(on_epoch_end=trainCallbackFunction)] if trainCallbackFunction != None else None
        self.__model.fit(x_train, y_train_categorical, epochs=epochsIn,validation_data=(x_val,y_valid_categorical),verbose=2, callbacks=callfunc)

        return self.__model
    
    def getAccuracy(self,x_test,y_test):
        y_test_categorical = keras.utils.to_categorical(y_test, self.__classcount)

        test_loss, test_accuracy = self.__model.evaluate(x_test, y_test_categorical)


        return (test_accuracy,test_loss)
    
    def predict(self,amplitude):
        
        if amplitude != numpy.ndarray:
            x_input = numpy.array([amplitude])
        else:
            x_input = amplitude

        y_pred = self.__model.predict(x_input)

        predicted_class = numpy.argmax(y_pred)

        confidence = y_pred[0, predicted_class]


        return (predicted_class,confidence)
    
    def saveModel(self):
        name = easygui.enterbox("Enter your Model name:", title="Model Name")

        if name == None:
            return

        self.__model.save(f"{name}.h5")
        

    def predict_LogisticRegression(self,X_test):
        filename = 'dataset.csv'
        if os.path.isfile(filename):
            return
        
        df = pd.read_csv(filename)

        X_in,y = df['data'].tolist(),df['classname']
        X_list = []
        for x in X_in:
            X_list.append(ast.literal_eval(x))
        
        X = numpy.array(X_list)
        y = numpy.array(y)

        # X_train,X_testSP,y_train,y_testSP = train_test_split(X,y,test_size = 0.5,random_state= random.randint(1,500))

        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        model = LogisticRegression(solver='lbfgs',max_iter=5000)

        # model.fit(X_train,y_train)
        model.fit(X,y)


        pred = model.predict(X_test)

        # tn,fp,fn,tp = metrics.confusion_matrix(y_testSP,pred).ravel()


        # accuracy = metrics.accuracy_score(y_testSP,pred)
        confidence = model.predict_proba(X_test)
        # print(f'score = {model.score(X_test,y_test)}')
        # print(f'accuracy = {metrics.accuracy_score(y_test,pred)}')


        return pred,confidence

    # sequential model
    # def get_SequentialModel(self,x_train,learning_rate = 0.01):

    #     self.model = keras.models.Sequential()
    #     self.model.add(keras.layers.Dense(100,activation='relu',input_shape=(x_train.shape[1],)))
    #     self.model.add(keras.layers.Dense(100,activation='relu'))
    #     self.model.add(keras.layers.Dense(1))
    #     self.model.compile(loss='mse',optimizer=keras.optimizers.SGD(learning_rate=learning_rate))

    #     return self.model



# plt.plot(X,Y,'.')
# plt.show()

# #Load Model
# model_2 = keras.models.load_model('medium.h5')

# # print(model_2.summary())
# print(model_2.get_weights())
