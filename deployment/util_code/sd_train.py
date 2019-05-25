#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:41:44 2019

@author: xiaodanchen
"""

from util_code import data_preprocessing
from util_code import lstm_train

def train_sd_model(data, glove_path, model_path):
    # here you should change the minimum speech length, the maximum speech length
    # the minimum word count and the maximum word count you use to control word frequency
    # e.g. here in order to run the model more quickly, for every speech, I only maintain speeches with length between 5 and 200 words
    # and maintain word tokens that appear in the whole corpus between 1000 and 80000 words
    # data = data_preprocessing.read_data('../data/data.csv')
    data['tagged'] = -1  # add tagged column
    train_corpus, test_corpus, train_labels, test_labels = data_preprocessing.split(data,5,200,1000,80000,
                                                                                   common_tokens_path='/Users/xiaodanchen/Desktop/0509/model/common_tokens.pickle')
    # here 200 refers to the input length of each speech into keras, you could change here
    # and define the path for saving model and pretrained glove word embedding
    model,_= lstm_train.train(train_corpus, test_corpus,
                              train_labels, test_labels,
                              200,
                              save_model_path=model_path + '/lstm_model.h5',
                              glove_path=glove_path,
                              tokenizer_path=model_path + '/tokenizer.pickle')
    return model