#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:51:09 2018

@author: jimmy
"""

import numpy as np
import pandas as pd
import sys

test_x = pd.read_csv(sys.argv[3])
w_T = np.load('./parameter_best/generative/weight.npy')
bias = np.load('./parameter_best/generative/bias.npy')

data_dim = test_x.shape[1]

data_iter = test_x.iterrows()
output_file = open(sys.argv[4],'w')
output_file.write('id,value\n')
while True:
    try:
        data = next(data_iter)
        data_id = data[0]
        data_value = np.reshape(data[1].values, (data_dim, 1))
    except:
        break
    ans=0
    z = np.matmul(w_T, data_value) + bias
    output_P = 1/(1+np.exp(-1*z))
    if output_P < 0.5:
        ans=1
    output_file.write("id_"+str(data_id)+','+str(ans)+'\n')
output_file.close()