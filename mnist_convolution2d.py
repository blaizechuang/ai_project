# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:02:56 2017

@author: user
"""
from keras.datasets import mnist
from matplotlib import pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import numpy as np
import keras

# Load image data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# local predefined data
log_filePath = 'TensorBoard/'


# reshape x to x_train_np, x_test_np
x_train_np = np.array(x_train)
x_train_np = x_train_np.reshape(x_train_np.shape[0], 1, 28, 28)

x_test_np = np.array(x_test)
x_test_np = x_test_np.reshape(x_test_np.shape[0], 1, 28, 28)

# convert type to float and limit the range to [0~1]
x_train_np = x_train_np.astype('float')
x_test_np = x_test.astype('float')
x_train_np /= 255
x_test_np /= 255

# reshape y to dimen 10
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# start model
model = Sequential()
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
# Comment out below below make the time elapsed from 24.58 min to 12.35 min
#model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# complie
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# keep log to show on tensorboard
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filePath, histogram_freq=1)
cbks = [tb_cb]

model.fit(x_train_np, y_train, 
          batch_size=32, nb_epoch=10, verbose=1, callbacks=cbks)


print(x_train.shape)
print(x_train_np.shape)
print(y_train.shape)
print(x_test.shape)
print(x_test_np.shape)
print(y_test.shape)
print(cbks)

plt.imshow(x_train[10])