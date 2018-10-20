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

train_x = train_x.drop(['EDUCATION'], axis = 1)

train_x = train_x.drop(['MARRIAGE'], axis = 1)


train_x = train_x.drop(['PAY_0'], axis = 1)

train_x = train_x.drop(['PAY_2'], axis = 1)

train_x = train_x.drop(['PAY_3'], axis = 1)

train_x = train_x.drop(['PAY_4'], axis = 1)

train_x = train_x.drop(['PAY_5'], axis = 1)

train_x = train_x.drop(['PAY_6'], axis = 1)


#train_x = pd.concat([train_x, pd.DataFrame(train_x['BILL_AMT1']/train_x['LIMIT_BAL'])], axis =1)



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
print(mean.shape)
'''
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
'''
mean = np.reshape(mean, (1, 14))
stddev = np.reshape(stddev, (1, 14))
mean_r = np.concatenate([mean]*20000, axis = 0)
stddev_r = np.concatenate([stddev]*20000, axis = 0)
train_x = (train_x - mean_r)/stddev_r
train_x = pd.concat([train_x, sex], axis =1)
train_x = pd.concat([train_x, education], axis =1)
train_x = pd.concat([train_x, marriage], axis =1)
train_x = pd.concat([train_x, pay_0], axis =1)
train_x = pd.concat([train_x, pay_2], axis =1)
train_x = pd.concat([train_x, pay_3], axis =1)
train_x = pd.concat([train_x, pay_4], axis =1)
train_x = pd.concat([train_x, pay_5], axis =1)
train_x = pd.concat([train_x, pay_6], axis =1)

data_iter = train_x.iterrows() #iterator: (id, pd.Series)
label_iter = train_y.iterrows() #iterator: (id, pd.Series)

num_unit = 35
while True:
    try:
        data = np.reshape(next(data_iter)[1].values, (93, 1))
        label = next(label_iter)[1].values[0]
    except:
        break
    #data = np.concatenate( (data, [[1]]*num_unit), axis = 0 )
    data_list.append( (data, label) )
#finish pre-processing
    
training_data = data_list[len(data_list)//10:]    
validation_data = data_list[:len(data_list)//10]      
data_dim = train_x.shape[1]
#model initialization

weight_decay = 0.99999

weight = np.random.randn(num_unit,1) #include bias
hidden = np.random.randn(data_dim,num_unit)
output_bias = 1
hidden_bias = np.ones((num_unit,1))
def sigmoid(x):
    return 1/(1+np.exp(-1*x)) 
#training
def train(epoch, batch, weight, hidden, output_bias, hidden_bias):
    
    learning_rate = 0.0004
    beta1 = 0.6
    beta2 = 0.9
    m_0b = 0
    v_0b = 0
    m_0 = np.zeros((num_unit,1))
    v_0 = np.zeros((num_unit,1))
    epsilon = np.ones((num_unit,1))* 0.00000001
    
    
    m_0hb =  np.zeros((num_unit,1))
    v_0hb =  np.zeros((num_unit,1))
    epsilon_hb = np.ones((num_unit,1))* 0.00000001
    m_0h = np.zeros((data_dim,num_unit))
    v_0h = np.zeros((data_dim,num_unit))
    epsilon_h = np.ones((data_dim,num_unit))* 0.00000001
    
    iteration = 0
    for epoch_num in range(epoch):
        np.random.shuffle(training_data)
        epoch_acc = 0
        epoch_loss = 0
        for batch_num in range(0, len(training_data), batch):
            iteration += 1
            #(feature , batch)
            batch_data = np.concatenate([i[0] for i in training_data[batch_num : batch_num+batch]], axis =1)
            batch_data = np.transpose(batch_data)
            batch_label = np.array([i[1] for i in training_data[batch_num : batch_num+batch]])
            batch_label = np.reshape(batch_label, (1, batch))
            
            #(num_unit, batch)
            hidden_out = np.matmul(np.transpose(hidden), batch_data.T)+hidden_bias
            hidden_out_activate = sigmoid(hidden_out)
            #(1, batch)
            Z = np.matmul(np.transpose(weight), hidden_out_activate)
            output =  sigmoid(Z+output_bias)
            
            cross_entropy = -1*(np.matmul(np.log(output), batch_label.T) + 
                                np.matmul(np.log(1-output), (1-batch_label).T ))
            cross_entropy /= batch
            #(1, num_unit)
            gradient1 = -1 * np.matmul((batch_label - output), np.transpose(hidden_out_activate) ) 
            grad_bias = np.sum( -1 * (batch_label - output) )/batch
            gradient1 = gradient1.T
            #(num , batch)
            output_diag = np.diag(np.reshape(batch_label - output, (-1,)))
            w_1 = np.concatenate([weight]*batch, axis = 1)
            
            #(num_unit, batch)
            gradient2 = -1 * np.matmul(w_1, output_diag) * hidden_out_activate*(1-hidden_out_activate) ###
            hidden_bias_grad = np.matmul(gradient2, np.array([[1]]*batch) )
            #(num, feature)
            gradient2 = np.matmul(gradient2, batch_data)
            hidden_grad = gradient2.T
            gradient1 /= batch
            hidden_grad /= batch
            
            m_0b = beta1*m_0b+(1-beta1)*grad_bias
            v_0b=beta2*v_0b+(1-beta2)*(grad_bias**2)
            m_0=beta1*m_0+(1-beta1)*gradient1
            v_0=beta2*v_0+(1-beta2)*(gradient1**2)
            
            mtb_hat=m_0b/(1-beta1**iteration)  
            vtb_hat=v_0b/(1-beta2**iteration)
            mt_hat=m_0/(1-beta1**iteration)   
            vt_hat=v_0/(1-beta2**iteration)
            
            m_0hb = beta1*m_0b+(1-beta1)*hidden_bias_grad
            v_0hb=beta2*v_0b+(1-beta2)*(hidden_bias_grad**2)
            m_0h=beta1*m_0h+(1-beta1)*hidden_grad
            v_0h=beta2*v_0h+(1-beta2)*(hidden_grad**2)
            mthb_hat=m_0hb/(1-beta1**iteration)  
            vthb_hat=v_0hb/(1-beta2**iteration)
            mth_hat=m_0h/(1-beta1**iteration)   
            vth_hat=v_0h/(1-beta2**iteration)
            #print( (mt_hat/(np.sqrt(vt_hat)+epsilon)).shape)
            output_bias = output_bias*weight_decay - learning_rate *(mtb_hat/(np.sqrt(vtb_hat)+0.00000001) )
            hidden_bias = hidden_bias*weight_decay - learning_rate *(mthb_hat/(np.sqrt(vthb_hat)+epsilon_hb) )
            weight = weight*weight_decay - learning_rate*(mt_hat/(np.sqrt(vt_hat)+epsilon) )
            hidden = hidden*weight_decay - learning_rate*(mth_hat/(np.sqrt(vth_hat)+epsilon_h) )
            '''
            weight = weight - learning_rate*(gradient1.T)
            hidden = hidden - learning_rate*(hidden_grad)
            '''
            epoch_loss += cross_entropy[0][0]
            #if p=0.5 -> class_0
            ans = (output >= np.array([[0.5]*batch])).astype(np.int8)
            
            acc = np.sum(ans == batch_label) / batch
            epoch_acc += acc
        print("loss: ", epoch_loss / (len(training_data)/batch))
        print("acc: ", epoch_acc / (len(training_data)/batch))
        val_loss = 0
        val_acc = 0
        for val_data in validation_data:
            hidden_out = np.matmul(np.transpose(hidden), val_data[0])+hidden_bias
            hidden_out_activate = sigmoid(hidden_out)
            Z = np.matmul(np.transpose(weight), hidden_out_activate)
            output =  sigmoid(Z+output_bias)
            
            cross_entropy = -1*((np.log(output)*val_data[1]) + 
                                np.log(1-output)*(1-val_data[1])) 
            val_loss += cross_entropy[0,0]
            
            ans = (output>=0.5).astype(np.int8)[0,0]
            val_acc += np.sum(ans == val_data[1])
        print("val_loss: ", val_loss / (len(validation_data)))
        print("val_acc: ", val_acc / (len(validation_data)))
            
    return weight, hidden, output_bias, hidden_bias

weight, hidden, output_bias, hidden_bias= train(300,72,weight, hidden, output_bias, hidden_bias)
np.save("./parameter/logistic/NN/weight.npy", weight)
np.save("./parameter/logistic/NN/hidden.npy", hidden)
np.save("./parameter/logistic/NN/output_bias.npy", output_bias)
np.save("./parameter/logistic/NN/hidden_bias.npy", hidden_bias)
np.save("./parameter/logistic/NN/mean.npy", mean)
np.save("./parameter/logistic/NN/stddev.npy", stddev)

