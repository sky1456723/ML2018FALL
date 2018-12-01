#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:03:14 2018

@author: jimmy
"""

import jieba
import torch
import torch.utils
import torch.nn as nn
import pandas as pd
import numpy as np
import gensim
import re
import Data
import Model

### DEVICE ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


print("Start cleaning Data")
jieba.load_userdict("./data/dict.txt.big")
train_file = open("./data/train_x.csv")
train_x = train_file.readlines()
train_file.close()
train_ans = open("./data/train_y.csv")
train_y = train_ans.readlines()
train_ans.close()
chinese_search = re.compile(u"[\u4e00-\u9fff]+")

wv = gensim.models.KeyedVectors.load("./word2vec/wv_26411.wv")

short_data = []
short_label = []
medium_data = []
medium_label = []
long_data = []
long_label = []
for id in range(1,len(train_x)):
    train_x[id] = train_x[id].replace("\n","")
    train_x[id] = train_x[id].split(",", maxsplit = 1)[1]
    train_y[id] = train_y[id].replace("\n","")
    train_y[id] = train_y[id].split(",", maxsplit = 1)[1]
    
    word_list = jieba.lcut(train_x[id])
    clean_list = []
    for word in word_list:
        check = chinese_search.match(word,0)
        if type(check)!= type(None):
            try:
                clean_list.append(wv[word])
            except:
                pad = np.array([0]*wv.vector_size)
                clean_list.append(pad)
    if len(clean_list) != 0:
        if len(clean_list) <= 30:
            short_data.append(np.array(clean_list))
            short_label.append(np.array([int(train_y[id])]))
        elif len(clean_list) > 30 and len(clean_list) <= 60:
            medium_data.append(np.array(clean_list))
            medium_label.append(np.array([int(train_y[id])]))
        elif len(clean_list) > 60 and len(clean_list) <= 100:
            long_data.append(np.array(clean_list))
            long_label.append(np.array([int(train_y[id])]))

#Finish data preprocessing

batch_size = 32
epoch = 100

short_dataset = Data.ChatbotDataset(data_x = short_data, data_y = short_label)
short_dataloader = torch.utils.data.DataLoader(dataset = short_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_fn,
                                         shuffle = True,
                                         num_workers = 1)

medium_dataset = Data.ChatbotDataset(data_x = medium_data, data_y = medium_label)
medium_dataloader = torch.utils.data.DataLoader(dataset = medium_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_fn,
                                         shuffle = True,
                                         num_workers = 1)

long_dataset = Data.ChatbotDataset(data_x = long_data, data_y = long_label)
long_dataloader = torch.utils.data.DataLoader(dataset = long_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_fn,
                                         shuffle = True,
                                         num_workers = 1)
#Finish data loading

wv_size = wv.vector_size
h_size = 256
n_layers = 2
dropout = 0
bi = True
dnn_units = [512,256,128]

model = Model.MLHW4(wv_size = wv_size, h_size = h_size,
                    n_layers = n_layers, dropout_rate = dropout,
                    bi = bi, dnn_units = dnn_units).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
print("Starting training")
print(optimizer)
print(device)
for e in range(epoch):
    epoch_loss = 0
    acc = 0
    print("Epoch: ",e+1)
    for data, label, length in short_dataloader:
        data = data.to(device)
        label = label.to(device)
        length = length.to(device)
        
        optimizer.zero_grad()
        pred = model(data, length)
        loss = criterion(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
    for data, label, length in medium_dataloader:
        data = data.to(device)
        label = label.to(device)
        length = length.to(device)
        
        optimizer.zero_grad()
        pred = model(data, length)
        loss = criterion(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
        
    for data, label, length in long_dataloader:
        data = data.to(device)
        label = label.to(device)
        length = length.to(device)
        
        optimizer.zero_grad()
        pred = model(data, length)
        loss = criterion(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
        
    acc /= (len(short_dataset) + len(medium_dataset)+len(long_dataset))
    epoch_loss /= (len(short_dataloader)  +
                   len(medium_dataloader) + len(long_dataloader))
    print("Loss: ",epoch_loss)
    print("Accuracy: ",acc)

torch.save(model,"./first_model.pkl")
        
