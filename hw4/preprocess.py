#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:05:55 2018

@author: jimmy
"""

import jieba
import emoji
import torch
import pandas as pd
import gensim
import re

print("Start cleaning Data")
jieba.load_userdict("./data/dict.txt.big")
train_file = open("./data/train_x.csv")
train_x = train_file.readlines()
train_file.close()
punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")
clean_data = []
for id in range(len(train_x)):
    train_x[id] = train_x[id].replace("\n","")
    train_x[id] = train_x[id].split(",", maxsplit = 1)[1]
    #train_x[id] = emoji.demojize(train_x[id])
    word_list = jieba.lcut(train_x[id])
    word_list = [emoji.demojize(i) for i in word_list]
    clean_list = []
    for word in word_list:
        check = punctuation_search.match(word,0)
        if type(check)== type(None):
            clean_list.append(word)
    if len(clean_list) != 0:
        clean_data.append(clean_list)

print("Start training word2vec")
word2vec_model = gensim.models.Word2Vec(clean_data, size=200, window=5,
                                        min_count=5,
                                        workers=3, iter=30)
word2vec_model.save("./word2vec_emoji.model")
word2vec_model.wv.save("./wv_emoji.wv")
    
    