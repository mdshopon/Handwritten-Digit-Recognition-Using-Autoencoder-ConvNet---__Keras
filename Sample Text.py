
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

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Dataset Creating
# Dataset Creating
input_img = Input(shape=(1, 32, 32))

x = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((3, 3), border_mode='same')(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((3, 3), border_mode='same')(x)
x = Convolution2D(32, 2, 2, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(32, 2, 2, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((3, 3))(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
x = UpSampling2D((3, 3))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

Train = []
Test = []
for i in range(0, 4200):
    Train.append(i % 10)
for i in range(4200, 6000):
    Test.append(i % 10)

y_train = np.asarray(Train)
y_test = np.asarray(Test)

Name = []
for filename in listdir("Train"):
    if filename.endswith(".bmp"):
        Name.append("Train/" + filename)
Name.sort()
X_train = np.array([np.array(Image.open(fname)) for fname in Name])
Name2 = []
for filename in listdir("Test"):
    if filename.endswith(".bmp"):
        Name2.append("Test/" + filename)
Name2.sort()
X_test = np.array([np.array(Image.open(fname)) for fname in Name2])

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 1, 32, 32))
X_test = np.reshape(X_test, (len(X_test), 1, 32, 32))





train_datagen = ImageDataGenerator(
        #rescale=1./255,
        #zca_whitening=True,
        #featurewise_std_normalization=True,
         rotation_range=0.3,
         width_shift_range=0.3,
      #   height_shift_range=0.2
        # shear_range=0.1,
        # shear_range=0.1,
         #zoom_range=0.1,
       #  channel_shift_range=0.1,
       #  cval=0.1
        # rotation_range=0.2,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
       #  zoom_range=0.2
        # fill_mode='nearest'
        )
train_datagen.fit(X_train, augment=True)
test_datagen = ImageDataGenerator(

)

train_generator = train_datagen.flow(
         X_train, X_train,
         batch_size=batch_size)
validation_generator = test_datagen.flow(
        X_test,X_test,
        batch_size=batch_size)

from keras.callbacks import TensorBoard

autoencoder.fit_generator(
        train_generator,
        samples_per_epoch=4200,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=1800,
        callbacks = [TensorBoard(log_dir='/tmp/encoder')])



score = autoencoder.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
# print('Parameters: ', model.count_params())
# print(model.summar