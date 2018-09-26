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
datalistPM25 = []
labellist=[]
wrong_ans = np.array([0]*18)
wrong_col = np.array([[0]]*18)
for i in range(len(data)//18):
    for j in range(16):
        if (i == len(data)//18-1 or (i+1)%20==0) and j == 15:
            continue
        if not (data[i*18:(i+1)*18,j:j+9] == wrong_col).all(0).any():
            if j!=15:
                if not (data[i*18:(i+1)*18,j+9] == wrong_ans).all(0):
                    datalist.append(data[i*18:(i+1)*18,j:j+9])
                    datalistPM25.append(data[i*18+9,j:j+9])
                    labellist.append(data[i*18+9,j+9])

            else:
                if not(data[(i+1)*18:(i+2)*18,0] == wrong_ans).all(0):
                    datalist.append(data[i*18:(i+1)*18,j:j+9])
                    datalistPM25.append(data[i*18+9,j:j+9])
                    labellist.append(data[(i+1)*18+9,0]) 
    if i!=len(data)/18-1 and (i+1)%20!=0:
        for k in range(16,24):
            concat1=data[(i)*18:(i+1)*18,k:]
            concat2=data[(i+1)*18:(i+2)*18,:k+9-24]
            concat_data = np.concatenate( (concat1,concat2), axis=1 )
            if not( (concat_data == wrong_col).all(0).any() or (data[i*18:(i+1)*18,k+9-24] == wrong_ans).all(0) ):
                datalist.append(concat_data)
                concat1 = data[i*18+9,k:]
                concat2 = data[(i+1)*18+9,:k+9-24]
                datalistPM25.append(np.concatenate( (concat1,concat2), axis=0 ))
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
    datalistPM25[i] = (datalistPM25[i] - feature_mean[9,:]) / feature_stddev[9,:]

#================
for i in range(len(datalist)):
    datalist[i] = np.concatenate((np.reshape(datalist[i], (162,)), [1.0]))
    datalistPM25[i] = np.concatenate((datalistPM25[i], [1.0]))
train_data = list(datalist[len(datalist)//10:])
train_label = list(labellist[len(datalist)//10:])
validation_data = list(datalist[:len(datalist)//10])
validation_label = list(labellist[:len(datalist)//10])

train_data9 = list(datalistPM25[len(datalist)//10:])
validation_data9 = list(datalistPM25[:len(datalist)//10])
#===============================
input_index = list(range(len(train_data)))

weight=(np.random.rand(18*9+1))
weight9 = (np.random.rand(9+1))


mean_square_error=0
error=0
iteration=0
learning_rate=0.001
regular = 0.0001 #now best 0.001
epoch=500
error_list=[]


beta_1=0.6
beta_2=0.9
m_0=np.zeros(18*9+1,dtype=np.float32)
v_0=np.zeros(18*9+1,dtype=np.float32)
epsilon=np.array([0.00000001]*(18*9+1))

m_09=np.zeros(9+1,dtype=np.float32)
v_09=np.zeros(9+1,dtype=np.float32)
epsilon9=np.array([0.00000001]*(9+1))

for i in range(epoch):
    input_index = np.random.permutation(input_index)
    for j in input_index:
        iteration += 1
        
        output=np.dot(weight,train_data[j])    
        error += (train_label[j]-output)**2 + 0.5*regular*np.dot(weight,weight)
        gradient=(2*(output-train_label[j])*train_data[j]) + regular*(weight-np.concatenate( (np.zeros(18*9), [weight[162]])) )
        
        
        m_0=beta_1*m_0+(1-beta_1)*gradient
        v_0=beta_2*v_0+(1-beta_2)*(gradient**2)
        mt_hat=m_0/(1-beta_1**iteration)   
        vt_hat=v_0/(1-beta_2**iteration)
        
        weight=weight-learning_rate*(mt_hat/(np.sqrt(vt_hat)+epsilon) )
        '''
        #only pm2.5
        output=np.dot(weight9,train_data9[j])    
        error += (train_label[j]-output)**2 + 0.5*regular*np.dot(weight9,weight9)
        gradient=(2*(output-train_label[j])*train_data9[j]) + regular*(weight9-np.concatenate( (np.zeros(9), [weight9[9]])) )
        
        
        m_09=beta_1*m_09+(1-beta_1)*gradient
        v_09=beta_2*v_09+(1-beta_2)*(gradient**2)
        mt_hat=m_09/(1-beta_1**iteration)   
        vt_hat=v_09/(1-beta_2**iteration)
        
        weight9=weight9-learning_rate*(mt_hat/(np.sqrt(vt_hat)+epsilon9) )
        '''
    for j in range(len(validation_data)):
        
        output=np.dot(weight, validation_data[j])#+bias[0]
        square_error=((validation_label[j]-output)**2) + 0.5*regular*np.dot(weight,weight)
        mean_square_error += square_error
        '''
        #only pm2.5
        output=np.dot(weight9, validation_data9[j])#+bias[0]
        square_error=((validation_label[j]-output)**2) + 0.5*regular*np.dot(weight9,weight9)
        mean_square_error += square_error
        '''
    mean_square_error = (mean_square_error/len(validation_data) )**0.5
    print("epoch: ",i)
    print("training error: ", (error/len(train_data))**0.5)
    print("validation error: ", mean_square_error)
    np.save("./weight.npy", weight9)
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

