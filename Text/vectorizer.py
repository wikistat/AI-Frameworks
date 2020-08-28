import collections
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher


class Vectorizer:

    def __init__(self, vectorizer_type, nb_hash=None):
        self.vectorizer_type = vectorizer_type
        self.nb_hash = nb_hash

    def vectorizer_train(self, df, columns='Description', nb_gram=1, binary=False):
        data_array = [line for line in df[columns].values]
        # Hashage
        if self.nb_hash is None:
            feathash = None
            if self.vectorizer_type == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, nb_gram))
                data_vec = vec.fit_transform(data_array)
            else:
                vec = CountVectorizer(binary=binary)
                data_vec = vec.fit_transform(data_array)
        else:
            data_dic_array = [collections.Counter(line.split(" ")) for line in data_array]
            feathash = FeatureHasher(self.nb_hash)
            data_hash = feathash.fit_transform(data_dic_array)

            if self.vectorizer_type == "tfidf":
                vec = TfidfTransformer()
                data_vec = vec.fit_transform(data_hash)
            else:
                vec = None
                data_vec = data_hash

        return vec, feathash, data_vec

    @staticmethod
    def apply_vectorizer(df, vec, feathash, columns='Description'):
        data_array = [line for line in df[columns].values]

        # Hashage
        if feathash is None:
            data_hash = data_array
        else:
            data_dic_array = [collections.Counter(line.split(" ")) for line in data_array]
            data_hash = feathash.transform(data_dic_array)

        if vec is None:
            data_vec = data_hash
        else:
            data_vec = vec.transform(data_hash)
        return data_vec

    def save_dataframe(self, data, name=""):
        sparse.save_npz("data/vec_%s_nb_hash_%s_vectorizer_%s" % (name, str(self.nb_hash), str(self.vectorizer_type)),
                        data)

    def load_dataframe(self, name=""):
        return sparse.load_npz(
            "data/vec_%s_nb_hash_%s_vectorizer_%s.npz" % (name, str(self.nb_hash), str(self.vectorizer_type)))


