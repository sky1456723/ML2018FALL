#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:03:14 2018

@author: jimmy
"""

import jieba
import emoji
import torch
import torch.utils
import torch.nn as nn
import pandas as pd
import numpy as np
import gensim
import re
import Data
import Model
import pickle

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
punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")

wv = gensim.models.KeyedVectors.load("./word2vec/wv_emoji.wv")
wv_matrix = wv.vectors
dummy_col = np.zeros(wv_matrix.shape[1]).reshape(1, -1)
dummy_index = wv_matrix.shape[0]
wv_matrix = np.vstack((wv_matrix, dummy_col))
print("Dict size: ",len(wv.vocab.keys()))
print("wv size: ",wv.vector_size)
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
    word_list = [emoji.demojize(i) for i in word_list]
    if len(word_list) ==0:
        pass
    clean_list = []
    pad_num = 0
    for word in word_list:
        check = punctuation_search.match(word,0)
        if type(check)== type(None):
            try:
                clean_list.append(np.array([wv.vocab[word].index]))
            except:
                pad_num+=1
                pad = np.array([dummy_index])
                clean_list.append(pad)
    if pad_num > 2:
        pass
    if len(clean_list) != 0:
        if len(clean_list) <= 30:
            short_data.append(np.array(clean_list))
            short_label.append(np.array([int(train_y[id])]))
        elif len(clean_list) > 30 and len(clean_list) <= 60:
            medium_data.append(np.array(clean_list))
            medium_label.append(np.array([int(train_y[id])]))
        elif len(clean_list) > 60:
            if len(clean_list) > 100:
                clean_list = clean_list[:100]
            long_data.append(np.array(clean_list))
            long_label.append(np.array([int(train_y[id])]))

train_short = short_data[len(short_data)//10:]
train_short_label = short_label[len(short_data)//10:]
train_medium = medium_data[len(medium_data)//10:]
train_medium_label = medium_label[len(medium_data)//10:]
train_long = long_data[len(long_data)//10:]
train_long_label = long_label[len(long_data)//10:]

#validation data
print("Split validation set")

val_short_data = short_data[:len(short_data)//10]
val_short_label = short_label[:len(short_data)//10]
val_medium_data = medium_data[:len(medium_data)//10]
val_medium_label = medium_label[:len(medium_data)//10]
val_long_data = long_data[:len(long_data)//10]
val_long_label = long_label[:len(long_data)//10]

del short_data, short_label, medium_data, medium_label, long_data, long_label

#Finish data preprocessing

batch_size = 64
epoch = 20
#training set
short_dataset = Data.ChatbotDataset(data_x = train_short,
                                    data_y = train_short_label)
short_dataloader = torch.utils.data.DataLoader(dataset = short_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_no_sort,
                                         shuffle = True,
                                         num_workers = 2)

medium_dataset = Data.ChatbotDataset(data_x = train_medium,
                                     data_y = train_medium_label)
medium_dataloader = torch.utils.data.DataLoader(dataset = medium_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_no_sort,
                                         shuffle = True,
                                         num_workers = 2)

long_dataset = Data.ChatbotDataset(data_x = train_long,
                                   data_y = train_long_label)
long_dataloader = torch.utils.data.DataLoader(dataset = long_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_no_sort,
                                         shuffle = True,
                                         num_workers = 2)
#validation set
val_short_dataset = Data.ChatbotDataset(data_x = val_short_data,
                                        data_y = val_short_label)
val_short_dataloader = torch.utils.data.DataLoader(dataset = val_short_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_no_sort,
                                         shuffle = True,
                                         num_workers = 1)

val_medium_dataset = Data.ChatbotDataset(data_x = val_medium_data,
                                         data_y = val_medium_label)
val_medium_dataloader = torch.utils.data.DataLoader(dataset = val_medium_dataset,
                                                    batch_size = batch_size,
                                                    collate_fn = Data.collate_no_sort,
                                                    shuffle = True,
                                                    num_workers = 1)

val_long_dataset = Data.ChatbotDataset(data_x = val_long_data,
                                       data_y = val_long_label)
val_long_dataloader = torch.utils.data.DataLoader(dataset = val_long_dataset,
                                         batch_size = batch_size,
                                         collate_fn = Data.collate_no_sort,
                                         shuffle = True,
                                         num_workers = 1)


#Finish data loading
'''
def criterion(pred,label):
    boundary = torch.zeros_like(pred).to(device)
    loss = torch.sum( torch.max(boundary, 1-label*pred) )/batch_size
    return loss
'''
criterion = nn.BCELoss()

wv_size = wv.vector_size
h_size = 200
n_layers = 2
dropout = 0.4
bi = True
dnn_units = [20,20]
dnn_drop = [0.3,0.3]
n_embedding = dummy_index
embedding_dim = wv_matrix.shape[1]

model = Model.MLHW4(wv_size = wv_size, h_size = h_size,
                    n_layers = n_layers, dropout_rate = dropout,
                    bi = bi, dnn_units = dnn_units,
                    dnn_dropout = dnn_drop,
                    num_embedding = n_embedding,
                    embedding_dim = embedding_dim).to(device)
model.init_embedding(torch.nn.Parameter(torch.Tensor(wv_matrix).to(device),
                                        requires_grad = True) )
model = model.train()
#model.change_rnn_init()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001,
                                momentum=0.1, weight_decay=0)
print("Starting training")
print(model)
print(optimizer)
print(device)
best_acc = 0
for e in range(epoch):
    epoch_loss = 0
    acc = 0
    val_epoch_loss = 0
    val_acc = 0
    print("Epoch: ",e+1)
    model = model.train()
        
    for data, label, length in short_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        #print(data.shape)
        optimizer.zero_grad()
        pred = model(data, length)
        #print(pred.shape)
        loss = criterion(pred, label)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.recurrent.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
    for data, label, length in long_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        
        optimizer.zero_grad()
        pred = model(data, length)
        loss = criterion(pred, label)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.recurrent.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
    for data, label, length in medium_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        
        optimizer.zero_grad()
        pred = model(data, length)
        loss = criterion(pred, label)
        loss.backward()
        #nn.utils.clip_grad_norm_(model.recurrent.parameters(), 5)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
        
    
        
    acc /= (len(short_dataset) + len(medium_dataset)+len(long_dataset))
    epoch_loss /= (len(short_dataloader)  +
                   len(medium_dataloader) + len(long_dataloader))
    
    ### validation ###
    model = model.eval()
    for data, label, length in val_short_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        
        pred = model(data, length)
        loss = criterion(pred, label)
        
        val_epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        val_acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
    for data, label, length in val_medium_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        
        pred = model(data, length)
        loss = criterion(pred, label)
        
        val_epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        val_acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
        
    for data, label, length in val_long_dataloader:
        data = data.to(device).long().squeeze(dim=-1)
        label = label.to(device)
        length = length.to(device)
        
        pred = model(data, length)
        loss = criterion(pred, label)
        
        val_epoch_loss += loss.item()
        pred_ans = (pred>0.5).to(torch.float32)
        val_acc += torch.sum(torch.eq(pred_ans,label)).to(torch.float32).item()
    
    val_acc /= (len(val_short_dataset) +
                len(val_medium_dataset) +
                len(val_long_dataset))
    val_epoch_loss /= (len(val_short_dataloader) +
                       len(val_medium_dataloader) +
                       len(val_long_dataloader))
    
    print("Loss: ",epoch_loss)
    print("Accuracy: ",acc)
    print("val_Loss: ",val_epoch_loss)
    print("val_Accuracy: ",val_acc)
    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model,"./GRU_bi_2.pkl")
        #torch.save(optimizer.state_dict(),"./small.optim")

print("Best Acc: ", best_acc)
#torch.save(model,"./small.pkl")
#torch.save(optimizer.state_dict(),"./small.optim")
        
