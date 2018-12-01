#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:11:10 2018

@author: jimmy
"""

import torch
import torch.utils.data
import numpy as np

class ChatbotDataset(torch.utils.data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of list
    input data shape : (data_num, seq_len, feature_dim)
    
    It can alse load from .npy files,
    the file should contains array of word_vectors of several sequence,
    please set from_file to True if you use it,
    and give the file path to data_x and data_y parameters.
    
    __len__ will return the # of data
    """
    def __init__(self, data_x, data_y, from_file = False):
        if from_file:
            self.data = np.load(data_x)
            self.label = np.load(data_y)
        else:
            self.data = data_x
            self.label = data_y
    def __getitem__(self, index):
        data_seq_len = len(self.data[index])
        return [self.data[index], self.label[index], data_seq_len]
    def __len__(self):
        return len(self.data)
    
"""   
In dataloader, when we iter in it,
It will execute __next__ of DataLoaderIter,
And __next__ will call Dataset[index] (__getitem__)
and get # of batch time to save the return in a list
"""
def collate_fn(batch_data):
    """
    This function will dynamically pad the sequence to the same length
    in a batch when using dataloader,
    and the length depends on the max sequence length in the batch.
    
    #It's no need to explicitly call this function. 
    #Only put it into dataloader
    input : batch_data should be a list of data
    data would be a list of [list/np.array, list/np.array, int, int]
    each list/np.array has shape like : (seq_len, word_vec_size/one_hot_size)
    
    return: tuple of 4 Tensor (x, y, data_seq_len, label_seq_len)
    shape of x : (batch_size, max_data_seq_len, word_vec_size) FloatTensor
    shape of y : (batch_size, max_label_seq_len, 1) FloatTensor
    #1 means the index in one-hot 
    shape of data_seq_len : (batch_size, 1) LongTensor
    shape of label_seq_len : (batch_size, 1) LongTensor
    
    x means input data, sorted by the
    y means label
    data_seq_len means the input data sequence length before padding
    data_seq_len means the label sequence length before padding
    
    Please use this function when use dataloader. (collate_fn = collate_fn)
    The dataloader should load ChatbotDataset. 
    (dataset = Object of ChatbotDataset)
    len(dataloader) will return how many batch it will generate
    """
    data_type = str(type(batch_data[0][0]))
    if data_type == "<class 'numpy.ndarray'>":
        #padding data
        word_vec_size = batch_data[0][0].shape[1]
        max_data_len = max(batch_data, key=lambda x:x[2])[2]
        for data_num in range(len(batch_data)):
            data_len = batch_data[data_num][2]
            if max_data_len - data_len > 0:
                pad = [[0]*word_vec_size]*(max_data_len - data_len)
                pad = np.array(pad)
                unpadded = batch_data[data_num][0]
                batch_data[data_num][0] = np.concatenate((unpadded, pad),
                                                         axis = 0)
        
        batch_data.sort(key = lambda x:x[2], reverse=True)
        batch_x = [torch.Tensor(data[0]) for data in batch_data]
        batch_x = torch.stack(batch_x) 
        batch_y = [torch.Tensor(data[1]) for data in batch_data]
        batch_y = torch.stack(batch_y) 
        
        data_seq_len = [torch.Tensor([data[2]]) for data in batch_data]
        data_seq_len = torch.stack(data_seq_len).long()    
        return batch_x, batch_y, data_seq_len
    elif data_type == "<class 'list'>":
        #padding data
        #batch[0][0] : list of shape (seq_len, word_vec_size) 
        #batch[0][1] : list of shape (seq_len, one_hot_size)
        
        word_vec_size = len(batch_data[0][0][0])
        max_data_len = max(batch_data, key=lambda x:x[2])[2]
        for data_num in range(len(batch_data)):
            data_len = batch_data[data_num][2]
            for i in range(max_data_len - data_len):
                batch_data[data_num][0].append([0]*word_vec_size)
        
        batch_data.sort(key = lambda x:x[2], reverse=True)
        batch_x = [torch.Tensor(data[0]) for data in batch_data]
        batch_x = torch.stack(batch_x) 
        batch_y = [torch.Tensor(data[1]) for data in batch_data]
        batch_y = torch.stack(batch_y) 
        
        data_seq_len = [torch.Tensor([data[2]]) for data in batch_data]
        data_seq_len = torch.stack(data_seq_len).long()    
        return batch_x, batch_y, data_seq_len, label_seq_len