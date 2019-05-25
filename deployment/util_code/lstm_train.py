#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:47:57 2019

@author: xiaodanchen
"""
import pandas as pd
import numpy as np
import pickle
import string
import re
import numpy as np
import os
from collections import Counter
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')

#sklearn
from sklearn.model_selection import train_test_split

# data preprocessing
import util_code.data_preprocessing as data_preprocessing

# nlp packages
from gensim.models import Word2Vec

# machine learning packages
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

# plot
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#keras
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from keras import models
from keras import layers
from keras.models import load_model


def keras_tokenizer(train,test, max_len):
    #tokenization
    corpus = train + test
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    train_sequences = tokenizer.texts_to_sequences(train)
    test_sequences = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_sequences, maxlen=max_len)
    test_padded = pad_sequences(test_sequences, maxlen=max_len)

    # vocab_size
    vocab_size = len(tokenizer.word_index) + 1
    return train_padded,test_padded,vocab_size,tokenizer

def glove_embedding(tokenizer,vocab_size, glove_path='/Users/xiaodanchen/Desktop/0509/GloVe/glove.6B.100d.txt'):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(glove_path) 
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def LSTM_model(vocab_size,embedding_matrix,max_len):
    glove_model = models.Sequential()
    glove_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len, trainable=False))
    glove_model.add(Dropout(0.2))
    glove_model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.2))
    glove_model.add(layers.Dense(1, activation='sigmoid'))
    #With the set_weights method we load the pre-trained embeddings in the Embedding layer (here layer 0). 
    #By setting the trainable attribute to False, we make sure not to change the pre-trained embeddings.
    glove_model.layers[0].set_weights([embedding_matrix])
    glove_model.layers[0].trainable = False
    glove_model.summary()
    #plot_model(glove_model,to_file='glove_lstm.png',show_shapes=True)
    glove_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return glove_model

def training(glove_model,train_padded,test_padded, train_labels,test_labels,
             save_model_path='/Users/xiaodanchen/Desktop/0509/model/lstm_model.h5'):
    glove_model.fit(train_padded, 
                    np.array(train_labels), 
                    epochs=10,
                    verbose=False,
                    validation_split=0.3)
    glove_model.save(save_model_path)
    print('model saved')
    loss, accuracy = glove_model.evaluate(train_padded, np.array(train_labels), verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = glove_model.evaluate(test_padded, np.array(test_labels), verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    return glove_model

def prediction(model,test,test_labels):
    # get F1 score
    predictions = model.predict(test)
    new_predictions = []
    for i in predictions.reshape(1,-1)[0]:
        if i > 0.5:
            new_predictions.append(1)
        else:
            new_predictions.append(0)
    dic=classification_report(np.array(test_labels),new_predictions,output_dict=True)
    f1=[round(dic['0']['f1-score'],2),round(dic['1']['f1-score'],2),round(dic['weighted avg']['f1-score'],2)]
    return predictions,new_predictions,f1
    

def plot_history(history,filename):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig('./graphs/' + filename)
    plt.legend()
    
def train(train,test,
          train_labels,test_labels,
          max_len,
          save_model_path='/Users/xiaodanchen/Desktop/0509/GloVe/glove.6B.100d.txt',
          glove_path='/Users/xiaodanchen/Desktop/0509/model/lstm_model.h5',
          tokenizer_path='/Users/xiaodanchen/Desktop/0509/model/tokenizer.pickle'):
    train_padded,test_padded,vocab_size,tokenizer = keras_tokenizer(train,test,max_len)
    # saving tokenizer
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    embedding_matrix = glove_embedding(tokenizer,vocab_size,glove_path)
    glove_model = LSTM_model(vocab_size,embedding_matrix,max_len)
    model = training(glove_model,train_padded,test_padded, train_labels,test_labels,save_model_path)
    #predictions, dic = prediction(model,test_padded)
    return model,tokenizer
    