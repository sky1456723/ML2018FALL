#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:28:20 2018

@author: jimmy
"""
import numpy as np
import pandas as pd
import sys

test_x = pd.read_csv(sys.argv[3])
output_weight = np.load('./parameter_best/NN/weight.npy')
output_bias = np.load('./parameter_best/NN/output_bias.npy')
hidden_weight = np.load('./parameter_best/NN/hidden.npy')
hidden_bias = np.load('./parameter_best/NN/hidden_bias.npy')
mean = np.load('./parameter_best/NN/mean.npy')
stddev = np.load('./parameter_best/NN/stddev.npy')

sex = pd.get_dummies(test_x['SEX'], prefix='gender')
education = pd.get_dummies(test_x['EDUCATION'], prefix='education')
marriage = pd.get_dummies(test_x['MARRIAGE'], prefix='marriage')
pay_0 = pd.get_dummies(pd.Categorical(test_x['PAY_0'], categories = list(range(-2,9,1))),
                       prefix='pay_0')
pay_2 = pd.get_dummies(pd.Categorical(test_x['PAY_2'], categories = list(range(-2,9,1))),
                       prefix='pay_2')
pay_3 = pd.get_dummies(pd.Categorical(test_x['PAY_3'], categories = list(range(-2,9,1))),
                       prefix='pay_3')
pay_4 = pd.get_dummies(pd.Categorical(test_x['PAY_4'], categories = list(range(-2,9,1))),
                       prefix='pay_4')
pay_5 = pd.get_dummies(pd.Categorical(test_x['PAY_5'], categories = list(range(-2,9,1))),
                       prefix='pay_5')
pay_6 = pd.get_dummies(pd.Categorical(test_x['PAY_6'], categories = list(range(-2,9,1))),
                       prefix='pay_6')

test_x = test_x.drop(['SEX'], axis = 1)

test_x = test_x.drop(['EDUCATION'], axis = 1)

test_x = test_x.drop(['MARRIAGE'], axis = 1)


test_x = test_x.drop(['PAY_0'], axis = 1)

test_x = test_x.drop(['PAY_2'], axis = 1)

test_x = test_x.drop(['PAY_3'], axis = 1)

test_x = test_x.drop(['PAY_4'], axis = 1)

test_x = test_x.drop(['PAY_5'], axis = 1)

test_x = test_x.drop(['PAY_6'], axis = 1)

data_list = []
mean_r = np.concatenate([mean]*10000, axis = 0)
stddev_r = np.concatenate([stddev]*10000, axis = 0)

test_x = (test_x - mean_r)/stddev_r
test_x = pd.concat([test_x, sex], axis =1)
test_x = pd.concat([test_x, education], axis =1)
test_x = pd.concat([test_x, marriage], axis =1)
test_x = pd.concat([test_x, pay_0], axis =1)
test_x = pd.concat([test_x, pay_2], axis =1)
test_x = pd.concat([test_x, pay_3], axis =1)
test_x = pd.concat([test_x, pay_4], axis =1)
test_x = pd.concat([test_x, pay_5], axis =1)
test_x = pd.concat([test_x, pay_6], axis =1)


data_iter = test_x.iterrows()

def sigmoid(x):
    return 1/(1+np.exp(-1*x)) 

file = open(sys.argv[4],'w')
file.write('id,value\n')
while True:
    try:
        data = next(data_iter)
        data_id = data[0]
        data_value = np.reshape(data[1].values, (93, 1))
    except:
        break
    hidden_out = np.matmul(np.transpose(hidden_weight), data_value)+hidden_bias
    hidden_out_activate = sigmoid(hidden_out)
    Z = np.matmul(np.transpose(output_weight), hidden_out_activate)
    output =  sigmoid(Z+output_bias)
    
    ans = (output>=0.5).astype(np.int8)[0,0]
    file.write("id_"+str(data_id)+','+str(ans)+'\n')
file.close()
    