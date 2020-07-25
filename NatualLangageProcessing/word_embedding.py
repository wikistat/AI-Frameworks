import time
import numpy as np
import gensim
from scipy import sparse


class WordEmbedding:

    def __init__(self, word_embedding_type, args):
        self.word_embedding_type = word_embedding_type
        self.args = args

    def train(self):
        ts = time.time()
        if self.word_embedding_type == "word2vec":
            model = gensim.models.Word2Vec(**self.args)
        te = time.time()
        return model, te-ts

    @staticmethod
    def get_features_mean(lines, model):
        features = [model[x] for x in lines if x in model]
        if features == []:
            fm = np.zeros(model.vector_size)
        else:
            fm = np.mean(features, axis=0)
        return fm

    @staticmethod
    def get_matrix_features_means(X, model):
        ts = time.time()
        X_embedded_ = list(map(lambda x: WordEmbedding.get_features_mean(x, model), X))
        X_embedded = np.vstack(X_embedded_)
        te = time.time()
        return X_embedded, te-ts



