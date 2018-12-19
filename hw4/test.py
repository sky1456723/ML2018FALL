#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 16:02:53 2018

@author: jimmy
"""

import jieba
import emoji
import torch
import torch.utils
import numpy as np
import gensim
import re
import Data
import Model
import sys

### DEVICE ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("Start cleaning Data")
jieba.load_userdict(sys.argv[2])
test_file = open(sys.argv[1],'r')
test_x = test_file.readlines()
test_file.close()
punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")

wv = gensim.models.KeyedVectors.load("./word2vec/wv_emoji.wv")
wv_matrix = wv.vectors
dummy_col = np.zeros(wv_matrix.shape[1]).reshape(1, -1)
dummy_index = wv_matrix.shape[0]
wv_matrix = np.vstack((wv_matrix, dummy_col))

model = torch.load("./model/GRU_bi_DNN_more.pkl").to(device)
model = model.eval()
model2 = torch.load("./model/GRU_bi_2.pkl").to(device)
model2 = model2.eval()
model3 = torch.load("./model/GRU_bi_3.pkl").to(device)
model3 = model3.eval()
testing_data = []
prediction = []
for id in range(1,len(test_x)):
    test_x[id] = test_x[id].replace("\n","")
    test_x[id] = test_x[id].split(",", maxsplit = 1)[1]
    
    word_list = jieba.lcut(test_x[id])
    word_list = [emoji.demojize(i) for i in word_list]
    clean_list = []
    for word in word_list:
        check = punctuation_search.match(word,0)
        if type(check)== type(None):
            try:
                clean_list.append(np.array([wv.vocab[word].index]))
            except:
                pad = np.array([dummy_index])
                clean_list.append(pad)
    if len(clean_list) == 0:
        clean_list.append(np.array([dummy_index]))
    elif len(clean_list)>100:
        clean_list = clean_list[:100]
    testing_data.append(np.array(clean_list))
dummy_label = torch.zeros(len(testing_data),1)
test_set = Data.ChatbotDataset(data_x = testing_data, data_y = dummy_label)
dataloader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=256,
                                         collate_fn=Data.collate_no_sort,
                                         num_workers=2,
                                         shuffle=False)
#index = 0
print("Start Prediction")

for data, dummy_label, length in dataloader:
    pred = model(data.to(device).long().squeeze(dim=-1), length.to(device))
    pred2 = model2(data.to(device).long().squeeze(dim=-1), length.to(device))
    pred3 = model3(data.to(device).long().squeeze(dim=-1), length.to(device))
    pred = (pred+pred2+pred3)/3
    pred = (pred>0.5)
    for b in range(pred.shape[0]):
        prediction.append(pred[b,0].item())

output_file = open(sys.argv[3],'w')
output_file.write("id,label\n")
for n in range(len(prediction)):
    output_file.write(str(n)+","+str(prediction[n])+'\n')
output_file.close()
