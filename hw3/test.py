#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 00:50:39 2018

@author: jimmy
"""


import keras
import keras.models
import keras.utils
import  pandas as pd
import numpy as np

file = pd.read_csv("test.csv")
data = file['feature']
data = data.str.split()
data = list(data)
for i in range(len(data)):
    data[i] = list(map(int,data[i]))
    
data = np.array(data).reshape((-1,48,48,1)) /255

    
model1 = keras.models.load_model("./model/my_cnn_single.h5")
model2 = keras.models.load_model("./model/lots_of_cnn_0.h5")
model3 = keras.models.load_model("./model/lots_of_cnn_1.h5")
model4 = keras.models.load_model("./model/lots_of_cnn_2.h5")
model5 = keras.models.load_model("./model/lots_of_cnn_3.h5")
model6 = keras.models.load_model("./model/lots_of_cnn_4.h5")
print("Starting prediction")
result1 = model1.predict(data)
result2 = model2.predict(data)
result3 = model3.predict(data)
result4 = model4.predict(data)
result5 = model5.predict(data)
result6 = model6.predict(data)
result = (result1+result2+result3+result4+result5+result6)/6
class_label = list(map(np.argmax,result))

newFile = open("answer_ens_s01234.csv","w")
newFile.write("id,label\n")
for i in range(len(class_label)):
    newFile.write(str(i)+","+str(class_label[i])+"\n")
newFile.close()