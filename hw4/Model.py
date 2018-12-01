#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:11:39 2018

@author: jimmy
"""

import torch
import torch.nn as nn

class MLHW4(torch.nn.Module):
    def __init__(self, wv_size, h_size, n_layers,
                 dropout_rate, bi, dnn_units):
        """
        wv_size: word2vec_sim             (int)
        h_size: hidden_size               (int)
        n_layers: # of LSTM layers        (int)
        dropout_rate: P of dropout        (float)
        bi: bidirectional or not          (bool)
        dnn_units: # of neurons of fc     (list of int, from bottom to top)
        """
        super(MLHW4, self).__init__()
        self.recurrent = nn.LSTM(input_size = wv_size,
                                       hidden_size = h_size,
                                       num_layers = n_layers,
                                       dropout = dropout_rate,
                                       bidirectional = bi)
        self.DNN = nn.ModuleList([nn.Linear((int(bi)+1)*h_size, dnn_units[0]),
                                   nn.BatchNorm1d(dnn_units[0]),
                                   nn.LeakyReLU()])
        for i in range(1,len(dnn_units)):
            last = dnn_units[i-1]
            self.DNN.append(nn.Linear(last, dnn_units[i]))
            self.DNN.append(nn.BatchNorm1d(dnn_units[i]))
            self.DNN.append(nn.LeakyReLU())
        self.DNN.append(nn.Linear(dnn_units[-1], 1))
        self.DNN.append(nn.Sigmoid())
    def forward(self, x, length):
        output, (h_n, c_n) = self.recurrent(x)
        to_DNN = []
        for i in range(output.shape[0]):
            to_DNN.append(output[i,length[i].item()-1])
        to_DNN = torch.stack(to_DNN)
        
        for layers in self.DNN:
            to_DNN = layers(to_DNN)
        return to_DNN
        
        
        