from __future__ import print_function
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
from os import listdir
from keras.models import load_model
from os.path import isfile, join
import PIL.ImageOps
import matplotlib.cm as cm
import numpy as np
from skimage import color
from skimage import io
import pickle
import cv
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard

batch_size = 128
nb_classes = 10
nb_epoch = 200
# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (5, 5)


def XINDIANTRAIN():
    Name = []
    for i in range(0,10):
        L=[]
        for filename in listdir("NewDGray/"+str(i)+"/"):
            if(filename.endswith(".tif")):
                L.append("NewDGray/"+str(i)+"/"+filename)
        L.sort()
        for i in range(0,len(L)):
            Name.append(L[i])
    return Name
def YINDIANTRAIN():
    Test=[]
    for i in range(0, 10):
        path = 'NewDGray/' + str(i)
        num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
        for j in range(0, num_files):
            Test.append(i)
    return Test

def XINDIANTEST():
    Name2=[]
    for i in range(0, 10):
        L = []
        for filename in listdir("NewDGray/test/" + str(i) + "/"):
            if (filename.endswith(".tif")):
                L.append("NewDGray/test/" + str(i) + "/" + filename)
        L.sort()
        for i in range(0, len(L)):
            Name2.append(L[i])
    return Name2

def YINDIANTEST():
    Test=[]
    for i in range(0,10):
        path = 'NewDGray/test/'+str(i)
        num_files = len([f for f in os.listdir(path)
                         if os.path.isfile(os.path.join(path, f))])
        for j in range(0,num_files):
            Test.append(i)
    return Test

def XCMATERTRAIN():

    Name = []
    for filename in listdir("Train"):
        if filename.endswith(".bmp"):
            Name.append("Train/" + filename)
    Name.sort()
    return Name

def YCMATERTRAIN():
    Train = []
    for i in range(0, 4200):
        Train.append(i % 10)
    return Train

def XCMATERTEST():
    Name2 = []

    for filename in listdir("Test"):
        if filename.endswith(".bmp"):
            Name2.append("Test/" + filename)
        Name2.sort()
    return Name2

def YCMATERTEST():
    Test = []
    for i in range(4200, 6000):
        Test.append(i % 10)
    return Test

Name2=XCMATERTEST()
Name=XCMATERTRAIN()


y_test=YCMATERTEST()
y_train=YCMATERTRAIN()


X_train = np.array([np.array(Image.open(fname)) for fname in Name])
X_test = np.array([np.array(Image.open(fname)) for fname in Name2])
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


print(X_train.shape)
print(X_test.shape)
model = Sequential()
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 32, 32), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Parameters: ', model.count_params())
print(model.summary())