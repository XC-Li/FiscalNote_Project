#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:55:34 2019

@author: xiaodanchen
"""

#keras
# from keras.preprocessing import sequence
# from keras.utils import to_categorical
# from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.models import Sequential
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import plot_model
# from keras import models
# from keras import layers
from keras.models import load_model

import util_code.data_preprocessing as data_preprocessing
import util_code.lstm_train as lstm_train


def evaluation(data,model_path,min_speech_len,max_speech_len,min_wc,max_wc,input_len):
    '''
    evaluate model
    Args: data --> pandas DataFrame
          model_path --> the path for the model
          min_speech_len --> the minimum length of speech that remain
          max_speech_len --> the maximum length of speech that remain
          min_wc --> the minimum word count you use to control word frequency
          max_wc --> the maximum word count you use to control word frequency
          input_len --> the input length of each speech using keras
    Return: the prediction probability, the predictions, f1 score(no stance,have stance,weighted average)
    '''
    # data = data_preprocessing.read_data(data_path)
    train_corpus, test_corpus, train_labels, test_labels = data_preprocessing.split(data,
                                                                                    min_speech_len,max_speech_len,
                                                                                    min_wc,max_wc)                                                                               
    model = load_model(model_path)
    train_padded,test_padded,vocab_size,tokenizer = lstm_train.keras_tokenizer(train_corpus, test_corpus,input_len)
    pred_prob,prediction, f1 = lstm_train.prediction(model,test_padded, test_labels)
    pred_prob = list(pred_prob.reshape(1,-1)[0])
    return pred_prob, prediction, f1

if __name__ == '__main__':
    data_path = '../data/data.csv'
    model_path = '../model/lstm_model.h5'
    min_speech_len = 5
    max_speech_len = 200
    min_wc = 1000
    max_wc = 80000
    input_len = 200
    pred_prob, prediction, f1 = retrain(data_path,model_path,min_speech_len,max_speech_len,min_wc,max_wc,input_len)
