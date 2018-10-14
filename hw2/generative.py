# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 13:19:00 2018

@author: jimmy
"""
import numpy as np
import pandas as pd

#process training data
train_x = pd.read_csv('./data/train_x.csv')
train_y = pd.read_csv('./data/train_y.csv')
'''
#withou one-hot
sex = pd.get_dummies(train_x['SEX'], prefix='gender')
education = pd.get_dummies(train_x['EDUCATION'], prefix='education')
marriage = pd.get_dummies(train_x['MARRIAGE'], prefix='marriage')
pay_0 = pd.get_dummies(pd.Categorical(train_x['PAY_0'], categories = list(range(-2,9,1))),
                       prefix='pay_0')
pay_2 = pd.get_dummies(pd.Categorical(train_x['PAY_2'], categories = list(range(-2,9,1))),
                       prefix='pay_2')
pay_3 = pd.get_dummies(pd.Categorical(train_x['PAY_3'], categories = list(range(-2,9,1))),
                       prefix='pay_3')
pay_4 = pd.get_dummies(pd.Categorical(train_x['PAY_4'], categories = list(range(-2,9,1))),
                       prefix='pay_4')
pay_5 = pd.get_dummies(pd.Categorical(train_x['PAY_5'], categories = list(range(-2,9,1))),
                       prefix='pay_5')
pay_6 = pd.get_dummies(pd.Categorical(train_x['PAY_6'], categories = list(range(-2,9,1))),
                       prefix='pay_6')

train_x = train_x.drop(['SEX'], axis = 1)
train_x = pd.concat([train_x, sex], axis =1)
train_x = train_x.drop(['EDUCATION'], axis = 1)
train_x = pd.concat([train_x, education], axis =1)
train_x = train_x.drop(['MARRIAGE'], axis = 1)
train_x = pd.concat([train_x, marriage], axis =1)
train_x = train_x.drop(['PAY_0'], axis = 1)
train_x = pd.concat([train_x, pay_0], axis =1)
train_x = train_x.drop(['PAY_2'], axis = 1)
train_x = pd.concat([train_x, pay_2], axis =1)
train_x = train_x.drop(['PAY_3'], axis = 1)
train_x = pd.concat([train_x, pay_3], axis =1)
train_x = train_x.drop(['PAY_4'], axis = 1)
train_x = pd.concat([train_x, pay_4], axis =1)
train_x = train_x.drop(['PAY_5'], axis = 1)
train_x = pd.concat([train_x, pay_5], axis =1)
train_x = train_x.drop(['PAY_6'], axis = 1)
train_x = pd.concat([train_x, pay_6], axis =1)
'''
#end process training data

data_dim = train_x.shape[1]

#model

class1_num = 0
class1_mean = np.zeros((data_dim, 1))
class1_variance = np.zeros((data_dim, data_dim))

class0_num = 0
class0_mean = np.zeros((data_dim, 1))
class0_variance = np.zeros((data_dim, data_dim))

#calculate mean
data_iter = train_x.iterrows() #iterator: (id, pd.Series)
label_iter = train_y.iterrows() #iterator: (id, pd.Series)

while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (data_dim, 1))
        label = next(label_iter)[1].values #(1,)
    except:
        break
    if label[0] == 0:
        class0_num += 1
        class0_mean += data
    else:
        class1_num += 1
        class1_mean += data

class0_mean /= class0_num
class1_mean /= class1_num

#calculate variance
data_iter = train_x.iterrows() #iterator: (id, pd.Series)
label_iter = train_y.iterrows() #iterator: (id, pd.Series)

while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (data_dim, 1))
        label = next(label_iter)[1].values #(1,)
    except:
        break
    if label[0] == 0:
        class0_variance +=np.matmul( (data - class0_mean), np.transpose(data - class0_mean))
    else:
        class1_variance +=np.matmul( (data - class1_mean), np.transpose(data - class1_mean))


#cancel the numerator and denominator when combine
same_variance = class0_variance/20000 + class1_variance/20000

#define model

def Z(c0_num, c0_mean, c1_num, c1_mean, var):
    var_inv = np.linalg.inv(var)
    w_T = np.transpose(c0_mean - c1_mean)
    w_T = np.matmul(w_T, var_inv)
    b1 = -0.5*np.matmul( np.matmul(np.transpose(c0_mean), var_inv), c0_mean)
    b2 = 0.5*np.matmul( np.matmul(np.transpose(c1_mean), var_inv), c1_mean)
    b3 = np.log(c0_num / c1_num)
    b = b1 + b2 + b3
    
    return (w_T, b)

#Saving model
    
(w_T, b) = Z(c0_num = class0_num, c0_mean = class0_mean,
             c1_num = class1_num, c1_mean = class1_mean, var = same_variance)

np.save('./parameter/generative/weight.npy', w_T)
np.save('./parameter/generative/bias.npy', b)



