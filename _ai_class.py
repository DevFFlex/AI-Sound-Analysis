
import pandas as pd
import numpy
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow import keras
import soundfile as sf
import librosa
import os
import json
from _data_manager import DataManager

class AI(DataManager):

    def __init__(self):
        super().__init__()
        self.__model = None
        self.__classcount = None
        
    def isModel(self):
        return self.__model         


    def train(self,x_train,y_train,class_count,epochsIn = 50):
        self.__classcount = class_count
        
        self.__model = None
        self.__model = keras.models.Sequential()
 
        self.__model.add(keras.layers.Dense(100, activation='relu', input_shape=(x_train.shape[1],)))
        self.__model.add(keras.layers.Dense(100, activation='relu'))
        self.__model.add(keras.layers.Dense(self.__classcount, activation='softmax'))

        self.__model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # ตั้งค่าฟังก์ชันสูญเสียและตัวปรับแบบเหมาะสม

        y_train_categorical = keras.utils.to_categorical(y_train, self.__classcount)

        self.__model.fit(x_train, y_train_categorical, epochs=epochsIn)

        print(self.__model.get_weights())


        return self.__model
    
    def getAccuracy(self,x_test,y_test):
        y_test_categorical = keras.utils.to_categorical(y_test, self.__classcount)
        test_loss, test_accuracy = self.__model.evaluate(x_test, y_test_categorical)

        print('Test Loss:', test_loss)
        print('Test Accuracy:', test_accuracy)

        return (test_accuracy,test_loss)


    
    def predict(self,amplitude):
        
        if amplitude != numpy.ndarray:
            x_input = numpy.array([amplitude])
        else:
            x_input = amplitude


        # ทำนาย (predict)
        y_pred = self.__model.predict(x_input)

        # # หาคลาสที่มีความน่าจะเป็นสูงสุด
        predicted_class = numpy.argmax(y_pred)

        # # ค่า confidence คือความน่าจะเป็นสูงสุด
        confidence = y_pred[0, predicted_class]


        return (predicted_class,confidence)

            

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
