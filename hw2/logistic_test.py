#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:02:45 2018

@author: jimmy
"""

import numpy as np
import pandas as pd
#import sys

test_x = pd.read_csv('./data/test_x.csv')
w_T = np.load('./parameter/logistic/weight.npy')
mean = np.load('./parameter/logistic/mean.npy')
stddev = np.load('./parameter/logistic/stddev.npy')

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
test_x = pd.concat([test_x, sex], axis =1)
test_x = test_x.drop(['EDUCATION'], axis = 1)
test_x = pd.concat([test_x, education], axis =1)
test_x = test_x.drop(['MARRIAGE'], axis = 1)
test_x = pd.concat([test_x, marriage], axis =1)
test_x = test_x.drop(['PAY_0'], axis = 1)
test_x = pd.concat([test_x, pay_0], axis =1)
test_x = test_x.drop(['PAY_2'], axis = 1)
test_x = pd.concat([test_x, pay_2], axis =1)
test_x = test_x.drop(['PAY_3'], axis = 1)
test_x = pd.concat([test_x, pay_3], axis =1)
test_x = test_x.drop(['PAY_4'], axis = 1)
test_x = pd.concat([test_x, pay_4], axis =1)
test_x = test_x.drop(['PAY_5'], axis = 1)
test_x = pd.concat([test_x, pay_5], axis =1)
test_x = test_x.drop(['PAY_6'], axis = 1)
test_x = pd.concat([test_x, pay_6], axis =1)


data_dim = test_x.shape[1]

data_iter = test_x.iterrows()
output_file = open('./result/logistic/1014_log.csv','w')
output_file.write('id,value\n')
while True:
    try:
        data = next(data_iter)
        data_id = data[0]
        data_value = np.reshape(data[1].values, (data_dim, 1))
    except:
        break
    
    to_divide = (data_value-mean)
    for i in range(len(to_divide)):
        if stddev[i][0] != 0:
            to_divide[i][0] /= stddev[i][0]
    data_value = np.concatenate( (to_divide, [[1]]), axis = 0 )
    
    ans=0
    z = np.matmul(np.transpose(w_T), data_value) 
    output_P = 1/(1+np.exp(-1*z))
    if output_P >= 0.5:
        ans=1
    output_file.write("id_"+str(data_id)+','+str(ans)+'\n')
output_file.close()