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

Name2=XCMATERTRAIN()+XCMATERTEST()
Name=XINDIANTRAIN()


y_test=YCMATERTRAIN()+YCMATERTEST()
y_train=YINDIANTRAIN()

print(Name2)
print(Name)
print(y_test)
print(y_train)

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


#j=3000
# for i in range(0, 9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(X_test[j], cmap=pyplot.get_cmap('gray'))
#     j=j+1
# pyplot.show()



print(X_train.shape)
print(X_test.shape)

#Autoencoder Part:

MODEL=load_model('IndianStat.h5')

#MODEL.save('IndianStat.h5')

score = MODEL.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
