#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:14:36 2019

@author: xiaodanchen
"""
import pandas as pd
# import numpy as np
import pickle
import string
import re
# import numpy as np
# import scipy
# import os
from collections import Counter
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings('ignore')

#sklearn
from sklearn.model_selection import train_test_split

# nltk for nlp
from nltk import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

#relabel model
from util_code.Regex_Stance_Detection import RuleBasedStanceDetection

def read_data(data_path):
    '''
    read csv file
    input: your csv file location 
    output: DataFrame
    '''
    data = pd.read_csv(data_path)
    return data

# def save_jpeg(path,name):
#     fn = '%s.jpeg'%name
#     plt.savefig(os.path.join(path,fn))
    
def tokenize_text(text):
    '''
    tokenization
    '''
    tokens = word_tokenize(text.lower())
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_stopwords(text):
    '''
    remove stop words
    '''
    stop = list(set(stopwords.words('english')))
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stop]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_special_characters(text):
    '''
    remove special characters: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    '''
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_non_alphabetic_characters(text):
    '''
    remove non-alphabetic characters and numbers
    '''
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if w.isalpha()]
    return ' '.join(tokens)

def remove_tokens_with_length(text,length):
    '''
    remove tokens with length less than or equal the input length
    '''
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if len(w)>length]
    return ' '.join(tokens)

def get_common_tokens(data,min_occurence,max_occurence):
    '''
    get the vocab of the whole corpus
    '''
    vocab = Counter()
    corpus = data['text']
    for speech in corpus:
        tokens = speech.split()
        vocab.update(tokens)
    # keep tokens with a min occurence
    min_occurence = min_occurence
    common_tokens = [k for k,c in vocab.items() if c > min_occurence or c < max_occurence]
    return common_tokens

def text_preprocessing(corpus):
    '''
    preprocess 
    '''
    new_corpus = []
    for text in corpus:  # remove tqdm for single speech
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        text = remove_non_alphabetic_characters(text)
        text = remove_tokens_with_length(text,1)
        #text = clean_corpus(data,text,5,90000)
        new_corpus.append(text)
    return new_corpus

def relabel_data(data):
    sd = RuleBasedStanceDetection()
    X,y = data['text'].values, data['tagged'].values
    X = text_preprocessing(X)
    for i in tqdm(range(len(X))):
        if y[i] == -1:
            y[i] = sd.stance_detection_labeler(X[i])
    return X,y

def change_labels(labels):
    # relabel '-1' to '0' for train_labels for later deep learning model
    for i in range(len(labels)):
        if labels[i]==-1:
            labels[i]=0   
    return labels

def split(data,min_len,max_len,min_occurence,max_occurence,common_tokens_path='/Users/xiaodanchen/Desktop/0509/model/common_tokens.pickle'):
    X,y = relabel_data(data)
    y = change_labels(y)
    X,y = get_fixed_length_range_data(X,y,min_len,max_len)
    X = clean_corpus(X,min_occurence,max_occurence,
                     common_tokens_path=common_tokens_path)
    train_corpus, test_corpus, train_labels, test_labels = train_test_split(X,y,test_size=0.3)
    return train_corpus, test_corpus, train_labels, test_labels

# remove speeches whose length is shorter than min_len or longer than max_len for training data
def get_fixed_length_range_data(corpus,labels,min_len,max_len):
    '''
    remove speeches whose length is shorter than min_len or longer than max_len 
    input: speeches
    output: target corpus, target labels
    '''
    new_corpus = []
    new_labels = []
    for i in range(len(corpus)):
        if len(corpus[i].split()) > min_len and len(corpus[i].split()) <= max_len :
            new_corpus.append(corpus[i])
            new_labels.append(labels[i])
        else:
            continue
    return new_corpus, new_labels

def get_common_tokens(corpus,min_occurence,max_occurence):
    '''
    get the vocab of the whole corpus
    '''
    vocab = Counter()
    for speech in corpus:
        tokens = speech.split()
        vocab.update(tokens)
    # keep tokens with a min occurence
    common_tokens = [k for k,c in vocab.items() if c > min_occurence and c < max_occurence]
    
    # new vocab
    new_dic = {}
    for k,c in vocab.items(): 
        if c > min_occurence and c < max_occurence:
            new_dic[k]=c
        
    return common_tokens

def clean_corpus(corpus,min_occurence,max_occurence,
                 common_tokens_path='/Users/xiaodanchen/Desktop/0509/model/common_tokens.pickle'):
    '''
    ensure all speeches in a corpus only keep tokens with a min occurence
    input: corpus, common tokens with a min occurence in the whole corpus
    output: new target corpus
    '''
    common_tokens = get_common_tokens(corpus,min_occurence,max_occurence)
    # saving common_tokens
    with open(common_tokens_path, 'wb') as f:
        pickle.dump(common_tokens,f, protocol=pickle.HIGHEST_PROTOCOL)

    new_corpus = []
    for i in tqdm(range(len(corpus))):
        tokens = corpus[i].split()
        tokens = [w for w in tokens if w in common_tokens]
        new_speech = ' '.join(tokens)
        new_corpus.append(new_speech)
    return new_corpus