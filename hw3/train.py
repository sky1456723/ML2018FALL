#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 23:25:33 2018

@author: jimmy
"""

import keras
import pandas as pd
import numpy as np
import keras.preprocessing
import keras.regularizers
import keras.initializers
import keras.optimizers
import os

data = pd.read_csv("train.csv")
x_train = data[:]['feature']
x_train = x_train.str.split()

x_train_data = list(x_train.values)
x_train_data2 = np.array(x_train_data,dtype=np.float32)
x_train_data2 = (x_train_data2.reshape((-1,48,48,1))) /255
y_train = data[:]['label']
y_train = list(y_train.values)

for i in range(len(y_train)):
    temp = np.zeros(7)
    temp[y_train[i]] = 1
    y_train[i] = temp
y_train = np.array(y_train)

training = x_train_data2[len(x_train_data2)//10:,:,:,:]
validation = x_train_data2[:len(x_train_data2)//10,:,:,:]
training_label = y_train[len(y_train)//10:,:]
validation_label = y_train[:len(y_train)//10,:]
augmentData = keras.preprocessing.image.ImageDataGenerator(zoom_range = 0.4, rotation_range = 50, shear_range = 0.2,
                                                           horizontal_flip=True, vertical_flip = False, fill_mode = "constant")
originData = keras.preprocessing.image.ImageDataGenerator()

model = keras.Sequential()
model.add(keras.layers.Conv2D(64,(3,3),input_shape=(48,48,1), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(64,(3,3), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(128,(3,3), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(128,(3,3), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(128,(3,3), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(128,(3,3), kernel_initializer = keras.initializers.he_normal(), padding='same') )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=400, kernel_initializer = keras.initializers.he_normal()) )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=256, kernel_initializer = keras.initializers.he_normal()) )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=256, kernel_initializer = keras.initializers.he_normal()) )
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(units=7, kernel_initializer = keras.initializers.he_normal()))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

import keras.callbacks


#early = keras.callbacks.EarlyStopping(patience = 50, monitor = 'val_acc')
history = model.fit_generator(augmentData.flow(training, training_label, batch_size = 100, seed = 0),
                   steps_per_epoch = len(training)//100,
                   validation_data = originData.flow(validation, validation_label, batch_size = 100,seed = 0),
                   validation_steps = len(validation)//100,
                   epochs = 200)#, callbacks = [early])

model.save("my_cnn.h5")