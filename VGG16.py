import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Conv2D,MaxPool2D,BatchNormalization,AveragePooling2D
from tensorflow.keras.datasets import mnist

input_shape=(28,28,1)

model=Sequential()

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))


model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dense(4096,activation='relu'))


model.add(Dense(10,activation='softmax'))