

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


batch_size = 128
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (5, 5)

#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Dataset Creating
#Dataset Creating

Train=[]
Test=[]
for i in range(0,4200):
    Train.append(i%10)
for i in range(4200,6000):
    Test.append(i%10)

y_train=np.asarray(Train)
y_test=np.asarray(Test)

Name = []
for filename in listdir("Train"):
  if filename.endswith(".bmp"):
       Name.append("Train/" + filename)
Name.sort()
X_train = np.array([np.array(Image.open(fname)) for fname in Name])
Name2=[]
for filename in listdir("Test"):
  if filename.endswith(".bmp"):
       Name2.append("Test/" + filename)
Name2.sort()
X_test= np.array([np.array(Image.open(fname)) for fname in Name2])


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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# print(Y_train)
# print(Y_test)


#datagen=ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#datagen.fit(X_train)
# os.makedirs('TrainAug')
# for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=4200, save_to_dir='TrainAug',  save_format='bmp'):
#     # create a grid of 3x3 images
#     for i in range(0, 9):
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i].reshape(32, 32), cmap=pyplot.get_cmap('gray'))
#     # show the plot
#     pyplot.show()
#     break
# datagen2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# datagen2.fit(X_test)
# os.makedirs('TestAug')
# for X_batch, y_batch in datagen2.flow(X_test, Y_test, batch_size=1800, save_to_dir='TestAug',
#                                      save_format='bmp'):
#     # create a grid of 3x3 images
#     for i in range(0, 9):
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(X_batch[i].reshape(32, 32), cmap=pyplot.get_cmap('gray'))
#     # show the plot
#     pyplot.show()
#     break
# for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=9):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(32, 32), cmap=pyplot.get_cmap('gray'))
# 	# show the plot
# 	pyplot.show()
# 	break
#
# X_batch, y_batch = datagen.flow(X_train, X_train, batch_size=32)

#fit_generator(datagen, samples_per_epoch=len(X_train), nb_epoch=100)
# model = Sequential()
# model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(nb_filters, 5, 5)) #Second kernel size 5,5 and first dropout 0.5
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# model = Sequential()
# model.add(Convolution2D(32, 5, 5, input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])
datagen = ImageDataGenerator(
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode='nearest'
        #featurewise_center=True,  # set input mean to 0 over the dataset
        #samplewise_center=False,  # set each sample mean to 0
       # featurewise_std_normalization=True,  # divide inputs by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
         #zca_whitening=True,  # apply ZCA whitening
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=False
        )
datagen.fit(X_train)

#datagen2=ImageDataGenerator()
#datagen2.fit(X_test)
model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_test, Y_test)
                    )

# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Parameters: ', model.count_params())
print(model.summary())

#99.34