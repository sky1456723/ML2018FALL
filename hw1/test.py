import sys
import numpy as np

test=open(sys.argv[1])
test_data=test.readlines()
#======================
for i in range(len(test_data)):
    test_data[i]=test_data[i].replace("\n","")
    test_data[i]=test_data[i].split(",")[2:]
    for j in range(len(test_data[i])):
        if test_data[i][j]=="NR":
            test_data[i][j]= 0.0
        else :
            try:
                test_data[i][j]= float(test_data[i][j])
            except ValueError:
                continue
#=======================
test_data = np.array(test_data)
#weight1 = np.load("./weight1.npy")
#weight2 = np.load("./weight2.npy")
#weight3 = np.load("./weight3.npy")
weight4 = np.load("./param_best/weight.npy")
#bias = np.load("./bias.npy")
feature_mean = np.load("./mean.npy")
feature_stddev = np.load("./stddev.npy")




file_out=open(sys.argv[2],'w')
file_out.write('id,value')
file_out.write('\n')
for i in range(len(test_data)//18):
    input = (test_data[i*18:(i+1)*18] - feature_mean) / (feature_stddev)
    input[1,:] = np.zeros((1,9))
    input[3,:] = np.zeros((1,9))
    input[13,:] = np.zeros((1,9))
    input = np.reshape(input, (162,))
    input = np.concatenate((input, [1]))
    #prediction1 = np.matmul(weight1, input ) #+ bias
    #prediction2 = np.matmul(weight2, input )
    #prediction3 = np.matmul(weight3, input )
    prediction4 = np.matmul(weight4, input )
    prediction = prediction4 #(4*prediction1 + 4*prediction2 + 1*prediction3) / (9)
    file_out.write('id_'+str(i)+','+str(prediction))
    file_out.write('\n')
file_out.close()




