"""By: Xiaochi (George) Li: github.com/XC-Li"""
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from scipy.sparse import hstack as sparse_hstack


class D2V(object):
    def __init__(self, file):
        self.model = Doc2Vec.load(file)

    def fit(self, X):
        pass

    def transform(self, X):
        temp = []
        for speech in X:
            temp.append(self.model.infer_vector(speech))
        return np.vstack(temp)


class StackedD2V(object):
    def __init__(self, file, vectorizer):
        self.d2v = Doc2Vec.load(file)
        self.vectorizer = vectorizer

    def fit(self, X):
        self.vectorizer.fit(X)

    def d2v_transform(self, X):
        temp = []
        for speech in X:
            temp.append(self.d2v.infer_vector(speech))
        return np.vstack(temp)

    def transform(self, X):
        bow = self.vectorizer.transform(X)
        d2v_emb = self.d2v_transform(X)
        combined_emb = sparse_hstack((bow, d2v_emb))
        return combined_emb
