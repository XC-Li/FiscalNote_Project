"""
Utility functions for preprocess
By: Xiaochi (George) Li: github.com/XC-Li
"""
import matplotlib.pyplot as plt
import spacy
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize

# use spacy to lemmatization
spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def spacy_lemma(speech):
    doc = spacy(speech)
    return " ".join([token.lemma_ for token in doc])


# remove numbers and punctuation
re_number_punctuation = re.compile('['+string.punctuation+']|\d')


def remove_number_punctuation(text):
    return re.sub(re_number_punctuation, '', text.lower())

# remove stopwords

nltk_stop_words = set(stopwords.words('english'))


def remove_stopwords(text, cutoff=10000):
    tokens = word_tokenize(remove_number_punctuation(text))
#     tokens = [token.strip() for token in tokens]  # no need for this line
    filtered_tokens = [token for token in tokens if token not in nltk_stop_words]
    filtered_text = ' '.join(filtered_tokens[:cutoff])
    return filtered_text


def remove_stopwords_return_list(text, cutoff=10000):
    tokens = word_tokenize(remove_number_punctuation(text))
    #     tokens = [token.strip() for token in tokens]  # no need for this line
    filtered_tokens = [token for token in tokens if token not in nltk_stop_words]
    return filtered_tokens[:cutoff]


def word_count_helper(text):
    tokens = word_tokenize(remove_number_punctuation(text))
    filtered_tokens = [token for token in tokens if token not in nltk_stop_words]
    return len(filtered_tokens)


def word_count(text_array):
    count = text_array.apply(word_count_helper)
    _ = plt.hist(count, bins=100)
    return count.describe()
