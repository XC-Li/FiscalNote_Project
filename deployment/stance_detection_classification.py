from util_code.corpus_loader import corpus_loader, untagged_corpus_loader  # Load data
from util_code.preprocess_utility import spacy_lemma, remove_stopwords  # Preprocess Function
from util_code.Pipeline_V1 import Pipeline  # Machine Learning Pipeline

from util_code.sd_train import train_sd_model  # train stance detection model
from keras.models import load_model  # load stance detection model
from util_code.sd_prediction import single_speech_prediction as StanceDetection  # stance detection prediction class

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# suppress warnings
import warnings
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore')


class StanceDetectionAndClassification(object):
    """
    Stance Detection and Stance Classification
    By: Xiaodan Chen(Stance Detection), Xiaochi Li(Stance Classification) github.com/XC-Li
    Mar 2019 - May 2019 @ FiscalNote
    Methods:
        __init__(self, data_path: str = '../opinion_mining/')
        retrain_stance_detection(self)
        retrain_stance_classification(self)
        stance_detection(self, speech: str)
        stance_classification(self, speech: str)
    """

    def __init__(self, data_path: str = '../opinion_mining/'):
        """
        Define the paths of data to retrain and trained model
        :param data_path: optional, the path to the data to retrain the models.
        """
        self.data_path = data_path
        self.model_path = './model'  # place of trained model
        self.pretrained_word2vec = './nlpword2vecembeddingspretrained/glove.6B.300d.txt'  # for mean embedding vectorizer
        self.sc_model = None  # stance classification model
        self.sd_model = None  # stance detection model
        self.load_model()  # load models
        self.glove_path = './nlpword2vecembeddingspretrained/glove.6B.100d.txt'  # for stance detection training

    def load_model(self) -> None:
        """
        Load the trained stance detection model and stance classification model,
        if not found, use data in `data_path` to retrain them
        Modify self.sc_model, self.sd_model
        """
        try:  # load stance classification model
            with open('./model/stance_classification.model', 'rb') as sc_file:
                self.sc_model = pickle.load(sc_file)
            print('Found Trained Stance Classification Model')
        except FileNotFoundError:
            print('Stance Classification Model Not Found, will retrain on data')
            self.retrain_stance_classification()
            with open('./model/stance_classification.model', 'wb') as sc_file:
                pickle.dump(self.sc_model, sc_file)

        try:  # load stance detection model
            self.sd_model = load_model('./model/lstm_model.h5')
            print('Found Trained Stance Detection Model')
        except FileNotFoundError:
            print('Stance Detection Model Not Found, will retrain on data')
            self.retrain_stance_detection(self.glove_path, './model/')

    def retrain_stance_detection(self, glove_path, model_path) -> None:  # for easy debug
        """Retrain the stance detection model
        modify self.sd_model
        glove_path:"""
        try:
            df = pd.read_pickle('./all_df.pickle', compression='zip')
        except FileNotFoundError:
            df = untagged_corpus_loader(tagged_df=None, path_root=self.data_path)
            df.to_pickle('./all_df.pickle', compression='zip')

        self.sd_model = train_sd_model(df, glove_path, model_path)

    def retrain_stance_classification(self) -> None:
        """Retrain the stance classification model,
        modify self.sc_model"""
        try:
            df = pd.read_pickle('./tagged_df.pickle', compression='zip')
        except FileNotFoundError:
            df = corpus_loader(data_root=self.data_path)
            df.to_pickle('./tagged_df.pickle', compression='zip')

        X = df['text']
        y = df['support']
        X = X.apply(remove_stopwords, cutoff=500)
        # X = X.apply(spacy_lemma)  # Optional: Lemmatization, very slow
        sc_model = LogisticRegression()
        tfidf = TfidfVectorizer()  # use tfidf as bag of words vectorizer
        # Optional: Stacked Word2Vec mean embedding, need pre-trained Word2Vec model
        # from util_code.mean_embedding_vectorizer import StackedEmbeddingVectorizer
        # stack_embedding_vectorizer = StackedEmbeddingVectorizer(tfidf)
        # stack_embedding_vectorizer.load(self.pretrained_word2vec)
        # ----------
        smote = SMOTE(random_state=42)  # smote over sampling
        self.sc_model = Pipeline(X, y,
                                 vectorizer=tfidf, model=sc_model, sampler=smote)
        self.sc_model.exec()
        print('Stance Classification Model Retrain Completed')

    def stance_detection(self, speech: str) -> float:
        """
        Stance Detection Model
        :param speech: string
        :return: float, the probability of the speech contain stance in it
        """
        min_speech_len = 5
        max_speech_len = 200
        min_wc = 1000
        max_wc = 80000
        input_len = 200
        data_path = './data/data.csv'
        model_path = './model/lstm_model.h5'
        glove_path = './GloVe/glove.6B.100d.txt'
        tokenizer_path = './model/tokenizer.pickle'
        common_tokens_path = './model/common_tokens.pickle'
        sd = StanceDetection(speech, data_path, model_path, glove_path,
                             tokenizer_path, common_tokens_path,
                             min_speech_len, max_speech_len,
                             min_wc, max_wc,
                             input_len)
        return sd.prediction()

    def stance_classification(self, speech: str) -> float:
        """
        stance classification model
        :param speech: string
        :return: float, the probability of the speech contain positive stance
        """
        if not isinstance(speech, str):
            raise TypeError('Input should be a speech (str)')
        speech = remove_stopwords(speech, cutoff=500)
        vector = self.sc_model.vectorizer.transform([speech])
        prediction = self.sc_model.model.predict_proba(vector)
        return prediction[0][1]
