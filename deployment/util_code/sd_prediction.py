#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:02:25 2019

@author: xiaodanchen
"""
#keras
# from keras.preprocessing import sequence
# from keras.utils import to_categorical
# from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.models import Sequential
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.utils import plot_model
# from keras import models
# from keras import layers
from keras.models import load_model

from util_code import data_preprocessing
from util_code import lstm_train
import pickle
import warnings 
warnings.filterwarnings('ignore')

class single_speech_prediction(object):
    def __init__(self,text,data_path,model_path,glove_path,tokenizer_path,common_tokens_path,
                 min_speech_len,max_speech_len,
                 min_wc,max_wc,
                 input_len):
        self.speech = text
        self.data_path = data_path
        self.model_path = model_path
        self.glove_path = glove_path
        self.tokenizer_path = tokenizer_path
        self.common_tokens_path = common_tokens_path
        self.min_speech_len = min_speech_len
        self.max_speech_len = max_speech_len
        self.min_wc = min_wc
        self.max_wc = max_wc
        self.input_len = input_len
        
 
    
    def get_pretrained_model(self):
        # loading
        model = load_model(self.model_path) #load model
        # load tokenizer
        with open(self.tokenizer_path, 'rb') as handle: 
            tokenizer = pickle.load(handle) 
        # load tokenizer
        with open(self.common_tokens_path, 'rb') as handle: 
            common_tokens = pickle.load(handle) 
            
        return model,tokenizer,common_tokens
        
    
    
    def text_preprocessing(self):
        # single speech preprocessing
        speech = []
        speech.append(self.speech)
        X= data_preprocessing.text_preprocessing(speech)
        X_new,_ = data_preprocessing.get_fixed_length_range_data(X,[1],
                                                                 self.min_speech_len,
                                                                 self.max_speech_len)
        # limit word frequency
        new_corpus = []
        model,tokenizer,common_tokens = self.get_pretrained_model()
        for i in range(len(X_new)):
            tokens = X_new[i].split()
            tokens = [w for w in tokens if w in common_tokens]
            new_speech = ' '.join(tokens)
            new_corpus.append(new_speech)
    
        # padding
        sequences = tokenizer.texts_to_sequences(new_corpus)
        padded = pad_sequences(sequences, maxlen=self.input_len)
        return padded,model
    
    def prediction(self):
        padded,model = self.text_preprocessing()     
        prediction = model.predict(padded)
        if len(prediction)==0:
            return -1
        else: 
            return prediction[0][0]
