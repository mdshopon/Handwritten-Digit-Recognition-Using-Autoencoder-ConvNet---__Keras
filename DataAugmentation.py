from __future__ import print_function
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

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(X_train[0].shape)
from matplotlib import pyplot

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
datagen = ImageDataGenerator( rotation_range=10,
          width_shift_range=0.1,
         height_shift_range=0.1
        )
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
	# create a grid of 3x3 images
	for i in range(0, 3):
		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(X_batch[i].reshape(32, 32), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break
# print(Y_train)
# print(Y_test)


# model = Sequential()
# model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
#                         border_mode='valid',
#                         input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(nb_filters, 5, 5))  # Second kernel size 5,5 and first dropout 0.5
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#
# model.add(Convolution2D(nb_filters+32, 2, 2))  # Second kernel size 5,5 and first dropout 0.5
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
#
# model.add(Convolution2D(nb_filters+96, 2, 2))  # Second kernel size 5,5 and first dropout 0.5
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('tanh'))
# model.add(Dropout(0.25))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
# model = Sequential()
# model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 32, 32), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(15, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='RMSprop',
#               metrics=['accuracy'])
#
#
# train_datagen = ImageDataGenerator(
#         #rescale=1./255,
#         #zca_whitening=True,
#         #featurewise_std_normalization=True,
#          rotation_range=0.3,
#          width_shift_range=0.2,
#       #   height_shift_range=0.2
#         # shear_range=0.1,
#         # shear_range=0.1,
#          #zoom_range=0.1,
#        #  channel_shift_range=0.1,
#        #  cval=0.1
#         # rotation_range=0.2,
#         # width_shift_range=0.2,
#         # height_shift_range=0.2,
#         # shear_range=0.2,
#        #  zoom_range=0.2
#         # fill_mode='nearest'
#         )
# train_datagen.fit(X_train, augment=True)
# test_datagen = ImageDataGenerator(
#
# )
#
# train_generator = train_datagen.flow(
#          X_train, Y_train,
#          batch_size=batch_size)
# validation_generator = test_datagen.flow(
#         X_test,Y_test,
#         batch_size=batch_size)
#
# from keras.callbacks import TensorBoard
#
# model.fit_generator(
#         train_generator,
#         samples_per_epoch=4200,
#         nb_epoch=1,
#         validation_data=validation_generator,
#         nb_val_samples=1800,
#         callbacks = [TensorBoard(log_dir='/tmp/model')])
#
#
# for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# 	# show the plot
# 	pyplot.show()
# 	break
#
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# # print('Parameters: ', model.count_params())
# # print(model.summar