#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:11:39 2018

@author: jimmy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLHW4(torch.nn.Module):
    def __init__(self, wv_size, h_size, n_layers,
                 dropout_rate, bi, dnn_units, dnn_dropout,
                 num_embedding, embedding_dim):
        """
        wv_size: word2vec_sim             (int)
        h_size: hidden_size               (int)
        n_layers: # of LSTM layers        (int)
        dropout_rate: P of dropout        (float)
        bi: bidirectional or not          (bool)
        dnn_units: # of neurons of fc     (list of int, from bottom to top)
        dnn_dropout: dropout of fc        (list of float, from bottom to top)
        """
        super(MLHW4, self).__init__()
        self.embedding = nn.Embedding(num_embedding, embedding_dim, padding_idx=-1)
        self.recurrent = nn.GRU(input_size = wv_size,
                                       hidden_size = h_size,
                                       num_layers = n_layers,
                                       dropout = dropout_rate,
                                       bidirectional = bi, batch_first=True)
        if dnn_units == None:
            self.DNN = nn.ModuleList([nn.Dropout(p=0.3),
                                      nn.Linear((int(bi)+1)*h_size, 1),
                                      nn.Sigmoid()])
        else:
            self.DNN = nn.ModuleList([nn.Dropout(p=0.5),
                                      nn.Linear((int(bi)+1)*h_size, dnn_units[0]),
                                      nn.BatchNorm1d(dnn_units[0]),
                                      nn.LeakyReLU(),
                                      nn.Dropout(p=dnn_dropout[0])])
            for i in range(1,len(dnn_units)):
                last = dnn_units[i-1]
                self.DNN.append(nn.Linear(last, dnn_units[i]))
                self.DNN.append(nn.BatchNorm1d(dnn_units[i]))
                self.DNN.append(nn.LeakyReLU())
                self.DNN.append(nn.Dropout(p=dnn_dropout[i]))
            self.DNN.append(nn.Linear(dnn_units[-1], 1))
            self.DNN.append(nn.Sigmoid())
    def forward(self, x, length):
        x = self.embedding(x)
        output, h_n = self.recurrent(x)
        to_DNN = []
        for i in range(output.shape[0]):
            to_DNN.append(output[i,length[i].item()-1])
        to_DNN = torch.stack(to_DNN)
        
        
        for layers in self.DNN:
            to_DNN = layers(to_DNN)
        
        return to_DNN
    def init_embedding(self, matrix):
        self.embedding.weight = matrix
        print(self.embedding.weight.requires_grad)
        
    def change_rnn_init(self):
        for name, matrix in self.recurrent.named_parameters():
            if "weight_hh" in name:
                h_size = matrix.shape[1]
                nn.init.orthogonal_(matrix[:h_size,:])
                nn.init.orthogonal_(matrix[h_size:2*h_size,:])
                nn.init.orthogonal_(matrix[2*h_size:3*h_size,:])
                nn.init.orthogonal_(matrix[3*h_size:4*h_size,:])
    
        
        