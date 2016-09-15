
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


input_img = Input(shape=(1, 32, 32))

x = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((3, 3), border_mode='same')(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(x)
x = MaxPooling2D((3, 3), border_mode='same')(x)
x = Convolution2D(32, 2, 2, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)



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

from keras.callbacks import TensorBoard
print(X_train.shape)
print(X_test.shape)


autoencoder.fit(X_train, X_train,
                nb_epoch=100,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks = [TensorBoard(log_dir='/tmp/autoencoder')]
                )


decoded_imgs = autoencoder.predict(X_test)


# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()