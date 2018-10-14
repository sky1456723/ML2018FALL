#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:16:48 2018

@author: jimmy
"""

import numpy as np
import pandas as pd

#process training data
train_x = pd.read_csv('./data/train_x.csv')
train_y = pd.read_csv('./data/train_y.csv')

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

#end process training data

data_dim = train_x.shape[1]

mean = np.zeros((data_dim, 1))
variance = np.zeros((data_dim, 1))

#calculate mean
data_iter = train_x.iterrows() #iterator: (id, pd.Series)

while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (data_dim, 1))
    except:
        break
    mean += data

mean /= 20000

#calculate variance
data_iter = train_x.iterrows() #iterator: (id, pd.Series)
while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (data_dim, 1))
    except:
        break
    variance += (data - mean)**2
variance /= 20000
stddev = np.sqrt(variance)

#normalization
data_list = []
data_iter = train_x.iterrows() #iterator: (id, pd.Series)
label_iter = train_y.iterrows() #iterator: (id, pd.Series)
while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (data_dim, 1))
        label = next(label_iter)[1].values[0]
    except:
        break
    to_divide = (data-mean)
    for i in range(len(to_divide)):
        if stddev[i][0] != 0:
            to_divide[i][0] /= stddev[i][0]
    to_divide = np.concatenate( (to_divide, [[1]]), axis = 0 )
    data_list.append( (to_divide, label) )
    
#finish pre-processing

#model initialization
weight = np.random.randn(94,1) #include bias


#training
def train(epoch, batch, weight):
    
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    m_0 = np.zeros((94,1))
    v_0 = np.zeros((94,1))
    epsilon = np.ones((94,1))* 0.00000001
    
    iteration = 0
    for epoch_num in range(epoch):
        np.random.shuffle(data_list)
        epoch_acc = 0
        epoch_loss = 0
        for batch_num in range(0, len(data_list), batch):
            iteration += 1
            batch_data = np.concatenate([i[0] for i in data_list[batch_num : batch_num+batch]], axis =1)
            batch_label = np.array([i[1] for i in data_list[batch_num : batch_num+batch]])
            batch_label = np.reshape(batch_label, (batch, 1))
            Z = np.matmul(np.transpose(weight), batch_data)
            output =  1/(1+np.exp(-1*Z)) 
            
            cross_entropy = -1*(np.matmul(np.log(output), batch_label) + 
                                np.matmul(np.log(output),(1-batch_label) ))
            cross_entropy /= batch
            gradient = -1 * np.matmul(batch_data, (batch_label - np.transpose(output) ) )
            gradient /= batch
            
            m_0=beta1*m_0+(1-beta1)*gradient
            v_0=beta2*v_0+(1-beta2)*(gradient**2)
            mt_hat=m_0/(1-beta1**iteration)   
            vt_hat=v_0/(1-beta2**iteration)
            
            weight=weight-learning_rate*(mt_hat/(np.sqrt(vt_hat)+epsilon) )
            
            epoch_loss += cross_entropy[0][0]
            #if p=0.5 -> class_0
            ans = (output > np.array([[0.5]*batch])).astype(np.int8)
            acc = np.sum(ans == np.transpose(batch_label)) / batch
            epoch_acc += acc
        print("loss: ", epoch_loss / (len(data_list)/batch))
        print("acc: ", epoch_acc / (len(data_list)/batch))
            
    return weight

weight = train(100,50,weight)

