import numpy as np
import sys
#============================
#============================
file=open("./train.csv",encoding = 'big5')
data=file.readlines()
file.close()
for i in range(len(data)):
    data[i]=data[i].replace("\n","")
    data[i]=data[i].split(",")[3:]
data.pop(0)
for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j]=="NR":
            data[i][j]= 0
        else :
            try:
                data[i][j]= float(data[i][j])
            except ValueError:
                continue
data=np.array(data,dtype=np.float32)
datalist=[]
labellist=[]
for i in range(len(data)//18):
    for j in range(16):
        if i == len(data)//18-1 and j == 15:
            continue
        datalist.append(data[i*18:(i+1)*18,j:j+9])
        if j != 15:
            labellist.append(data[i*18+9,j+9])
        else:
            labellist.append(data[(i+1)*18+9,0])
    if i!=len(data)/18-1:
        for k in range(16,24):
            concat1=data[(i)*18:(i+1)*18,k:]
            concat2=data[(i+1)*18:(i+2)*18,:k+9-24]
            datalist.append(np.concatenate( (concat1,concat2), axis=1 ))
            labellist.append(data[(i+1)*18+9,k+9-24])

feature_mean=np.array(datalist[0])

#=============================

for i in range(1,len(datalist)):
    feature_mean += datalist[i]
feature_mean = feature_mean/len(datalist)

#====================================
feature_stddev=(datalist[0]-feature_mean)**2
for i in range(1,len(datalist)):
    feature_stddev += (datalist[i]-feature_mean)**2
feature_stddev = np.sqrt(feature_stddev/len(datalist))

#======================================

for i in range(len(datalist)):
    datalist[i] = (datalist[i] - feature_mean) / feature_stddev 

#================

for i in range(len(datalist)):
    datalist[i] = np.concatenate((np.reshape(datalist[i], (162,)), [1.0]))
    

train_data = list(datalist[len(datalist)//10:])
train_label = list(labellist[len(datalist)//10:])
validation_data = list(datalist[:len(datalist)//10])
validation_label = list(labellist[:len(datalist)//10])
#===============================
input_index = list(range(len(train_data)))
weight=(np.random.rand(18*9+1))
#bias=np.random.rand(1)
#bias/=1
mean_square_error=0
error=0
iteration=0
learning_rate=0.002
regular = 0
epoch=500
error_list=[]


beta_1=0.6
beta_2=0.9
m_0=np.zeros(18*9+1,dtype=np.float32)
m_0_b=0
v_0=np.zeros(18*9+1,dtype=np.float32)
v_0_b=0
epsilon=np.array([0.00000001]*(18*9+1))

for i in range(epoch):
    input_index = np.random.permutation(input_index)
    for j in input_index:
        iteration += 1
        output=np.dot(weight,train_data[j])#+bias[0]    
        error += (train_label[j]-output)**2 + 0.5*regular*np.dot(weight,weight)
        gradient=(2*(output-train_label[j])*train_data[j]) + regular*(weight-np.concatenate( (np.zeros(18*9), [weight[162]])) )
        #gradient_b=2*(output-train_label[j]) 
        
        
        m_0=beta_1*m_0+(1-beta_1)*gradient
        #m_0_b=beta_1*m_0_b+(1-beta_1)*gradient_b
        v_0=beta_2*v_0+(1-beta_2)*(gradient**2)
        #v_0_b=beta_2*v_0_b+(1-beta_2)*(gradient_b**2)    
        mt_hat=m_0/(1-beta_1**iteration)   
        #mt_b_hat=m_0_b/(1-beta_1**iteration)
        vt_hat=v_0/(1-beta_2**iteration)
        #vt_b_hat=v_0_b/(1-beta_2**iteration)
        
        weight=weight-learning_rate*(mt_hat/(np.sqrt(vt_hat)+epsilon) )
        #bias=bias-learning_rate*(mt_b_hat/(vt_b_hat**0.5+0.00000001))
        '''
        weight = weight - learning_rate*gradient
        #bias = bias - learning_rate*gradient_b
        '''
    #if(i%400==0):learning_rate*=0.66
    for j in range(len(validation_data)):
        output=np.dot(weight, validation_data[j])#+bias[0]
        square_error=((validation_label[j]-output)**2) + 0.5*regular*np.dot(weight,weight)
        mean_square_error += square_error
    mean_square_error = (mean_square_error/len(validation_data) )**0.5
    print("epoch: ",i)
    print("training error: ", (error/len(train_data))**0.5)
    print("validation error: ", mean_square_error)
    np.save("./weight.npy", weight)
    #np.save("./bias.npy", bias)
    error_list.append(mean_square_error)
    error=0
    mean_square_error=0

error_list = np.array(error_list)

np.save("./weight.npy", weight)
#np.save("./bias.npy", bias)
np.save("./error.npy",error_list)
np.save("./mean.npy", feature_mean)
np.save("./stddev.npy", feature_stddev)

#================================

