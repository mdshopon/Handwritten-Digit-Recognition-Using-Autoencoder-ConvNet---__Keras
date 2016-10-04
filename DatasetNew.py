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


import cv2
import numpy as np


def invertBackground():
    for i in range(0,10):
        for filename in listdir("NewD/test/"+str(i)):
            if filename.endswith(".tif"):
                FILE="NewD/test/"+str(i)+"/"+filename
                print(FILE)
                image = Image.open(FILE)
                if(image.mode!='P'):
                    inverted_image = PIL.ImageOps.invert(image)
                    if filename.endswith('.tif'):
                        filename = filename[:-4]
                    FILE2 = "NewDInvert/test/"+str(i)+"/" + filename + ".tif"
                    inverted_image.save(FILE2)
                #print(FILE2)
            else:
                continue

def toGray():
    for i in range(0, 10):
        for filename in listdir("NewDInvert/test/"+str(i)):
            if filename.endswith(".tif"):
                FILE = "NewDInvert/test/"+str(i)+"/"+ filename;
                img = cv2.imread(FILE)
                col_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                image = cv2.resize(col_img, (32, 32))
                FILE = "NewDGray/test/"+str(i)+"/" + filename
                cv2.imwrite(FILE, image)
               # print(col_img.shape)
            else:
                continue
#invertBackground()
#toGray()
import os.path
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


Train = []
Test = []
for i in range(0, 4200):
    Train.append(i % 10)
for i in range(4200, 6000):
    Test.append(i % 10)
for i in range(0,10):
    path = 'NewDGray/'+str(i)
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
    for j in range(0,num_files):
        Train.append(i)

for i in range(0,10):
    path = 'NewDGray/test/'+str(i)
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
    for j in range(0,num_files):
        Test.append(i)



y_train = np.asarray(Train)
y_test = np.asarray(Test)

Name = []
for filename in listdir("Train"):
    if filename.endswith(".bmp"):
        Name.append("Train/" + filename)
Name.sort()
for i in range(0,10):
    L=[]
    for filename in listdir("NewDGray/"+str(i)+"/"):
        #rint(filename)
        if(filename.endswith(".tif")):
            L.append("NewDGray/"+str(i)+"/"+filename)
    L.sort()
    for i in range(0,len(L)):
        Name.append(L[i])
X_train = np.array([np.array(Image.open(fname)) for fname in Name])
Name2 = []

for filename in listdir("Test"):
    if filename.endswith(".bmp"):
        Name2.append("Test/" + filename)
Name2.sort()
for i in range(0,10):
    L=[]
    for filename in listdir("NewDGray/test/"+str(i)+"/"):
        if(filename.endswith(".tif")):
            L.append("NewDGray/test/"+str(i)+"/"+filename)
    L.sort()
    for i in range(0,len(L)):
        Name2.append(L[i])
    #Name2.append(L)
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