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
import keras.layers
import keras.optimizers
import os
import sys

data = pd.read_csv(sys.argv[1])
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
augmentData = keras.preprocessing.image.ImageDataGenerator(zoom_range = 0.4, rotation_range = 30, shear_range = 0.2,
                                                           horizontal_flip=True, vertical_flip = False, fill_mode = "constant")
originData = keras.preprocessing.image.ImageDataGenerator()

#model = keras.Sequential()
input_image = keras.layers.Input(shape=(48,48,1))
input1 = keras.layers.Conv2D(64,(3,3), padding='same')(input_image)
input1 = keras.layers.BatchNormalization()(input1)
input1 = keras.layers.PReLU()(input1)
input1 = keras.layers.Conv2D(64,(3,3), padding='same')(input1)
input1 = keras.layers.BatchNormalization()(input1)
input1 = keras.layers.PReLU()(input1)
input1 = keras.layers.MaxPooling2D((2,2))(input1)
input1 = keras.layers.Dropout(0.3)(input1)
input2 = keras.layers.Conv2D(128,(3,3), padding='same')(input1)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.Conv2D(128,(3,3), padding='same')(input2)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.MaxPooling2D((2,2))(input2)
input2 = keras.layers.Dropout(0.3)(input2)
input2 = keras.layers.Conv2D(128,(3,3), padding='same')(input2)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.Conv2D(128,(3,3), padding='same')(input2)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.Dropout(0.3)(input2)
input2 = keras.layers.MaxPooling2D((2,2))(input2)

input2 = keras.layers.Flatten()(input2)
input2 = keras.layers.Dense(units=512 )(input2)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.Dropout(0.4)(input2)
input2 = keras.layers.Dense(units=256 )(input2)
input2 = keras.layers.BatchNormalization()(input2)
input2 = keras.layers.PReLU()(input2)
input2 = keras.layers.Dropout(0.4)(input2)
input2 = keras.layers.Dense(units=7 )(input2)
out = keras.layers.Activation('softmax')(input2)

model = keras.models.Model(inputs = input_image, outputs = out)

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

import keras.callbacks


#early = keras.callbacks.EarlyStopping(patience = 50, monitor = 'val_acc')
history = model.fit_generator(augmentData.flow(training, training_label, batch_size = 100, seed = 0),
                   steps_per_epoch = len(training)//100,
                   validation_data = originData.flow(validation, validation_label, batch_size = 100,seed = 0),
                   validation_steps = len(validation)//100,
                   epochs = 200)#, callbacks = [early])

model.save("./model/cnn_check.h5")
