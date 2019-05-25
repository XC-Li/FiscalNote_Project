"""
Mean Embedding Vectorizer
Original code by Vlad, Improved by Xiaochi (George) Li github.com/XC-Li
Can fit in Pipeline_V1 directly as a vectorizer
"""

import gensim
import numpy as np
from numpy import hstack as np_hstack
from scipy.sparse import hstack as sparse_hstack # need this hstack to stack sparse matrix
from tqdm import tqdm_notebook as tqdm
from sklearn.decomposition import TruncatedSVD


class EmbeddingMatrix(object):
    """
    An implementation to replace Gensim Word2vec Object
    """
    def __init__(self):
        self.vocab = dict()

    def word_vec(self, w):
        return self.vocab[w]

    def add_word_vec(self, w, vec):
        self.vocab[w] = vec


class MeanEmbeddingVectorizer(object):
    """
    Mean Embedding of an article
    """
    def __init__(self, vectorizer=None):
        self.word2vec = None
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = None
        self.svd = None
        self.pca = False
        self.vectorizer = vectorizer

    def load(self, file):
        try:
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)
            self.word2vec = word2vec
            self.dim = word2vec.vector_size
        except ValueError:
            self.word2vec = EmbeddingMatrix()
            with open(file, encoding='utf-8') as f:
                raw_line = f.readline()
                self.dim = len(raw_line.split()) - 1
                while raw_line:
                    line = raw_line.split()
                    # specify dtype to avoid cannot perform reduce with flexible type
                    self.word2vec.add_word_vec(line[0], np.array(line[1:], np.float16))
                    raw_line = f.readline()

    def fit(self, X):
        pass

    def transform(self, X):
        return self.mev_transform(X)

    def mev_transform(self, X):
        return np.array([
            np.mean([self.word2vec.word_vec(w) for w in words if w in self.word2vec.vocab]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def transform_with_progress(self, X):
        result = []
        for words in tqdm(X):
            result.append(np.mean([self.word2vec.word_vec(w) for w in words if w in self.word2vec.vocab]
                                  or [np.zeros(self.dim)], axis=0))

        return np.array(result)


class StackedEmbeddingVectorizer(MeanEmbeddingVectorizer):

    def fit(self, X):
        self.vectorizer.fit(X)
        bow_shape = len(self.vectorizer.get_feature_names())
        # print(bow_shape, end='')
        if self.pca:  # use PCA to reduce tfidf dimension
            # print('reduce bow dimension', end='')
            X_vec = self.vectorizer.transform(X)
            self.svd = TruncatedSVD(n_components=self.dim, random_state=42)
            self.svd.fit(X_vec)

    def set_pca(self, pca):
        self.pca = pca

    def set_vectorizer(self, v):
        self.vectorizer = v

    def transform(self, X):
        bow = self.vectorizer.transform(X)
        if self.pca:
            bow = self.svd.transform(bow)

        # print(bow.shape[1], end='')
        mean_emb = self.mev_transform(X)
        # print(mean_emb.shape)
        if self.pca:  # use np.hstack for numpy array
            combined_emb = np_hstack((bow, mean_emb))
        else:  # use scipy.sparse.hstack for sparse array
            combined_emb = sparse_hstack((bow, mean_emb))
        return combined_emb


# mev = MeanEmbeddingVectorizer(vec_model)
# x = vectorizer.fit_transform(X)
# x_emd = mev.transform(X)
# x_count_comb = hstack((x,x_emd))
#
# # Run model with only vectorized text
# train_test_clf(x, Y)
# # Run model with only average embeddings
# train_test_clf(x_emd, Y)
# # Run model with stacked vectorized text and average embeddings
# train_test_clf(x_count_comb, Y)
