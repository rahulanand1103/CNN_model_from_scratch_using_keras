import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization
from tensorflow.keras.datasets import mnist

#example input
input_shape=(28,28,1)

model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(5,5),activation='relu',input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(2,2),activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()